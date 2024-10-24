import copy
import json
import os
import torch

# from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset


def triplets_to_json(triplets):
    triplets_dicts = []
    for idx, triplet in enumerate(triplets):
        triplets_dicts.append({
            "gene": triplet[0],
            "disease": triplet[1],
            "relation": triplet[2]
        })
    output = json.dumps(triplets_dicts)
    return output

POSITIVE_WEIGHT = (153 + 317) / (153 * 2)
NEGATIVE_WEIGHT = (153 + 317) / (317 * 2)
REMOVE_ALL_NEGATIVES = True
# INSTRUCTION = """Given a text, extract the gene-disease-relation triplets in a json format."""
INSTRUCTION = """**Extract triplets**: Identify and extract sets of three linked entities:
   - **Gene**: A human gene name, symbol (e.g., *SLC02A1*, *PCSK5*) or synonym.
   - **Human Disease**: A specific human disease or disorder name (e.g., *lung adenocarcinoma*, *coronary artery disease*).
   - **Relation**: The relationship between the gene and the human disease. These relation types of interest are *pathological role*, *causative activation*, *causative inhibition*, *causative mutation*, *modulator decrease disease*, *modulator increase disease*, *biomarker*, *associated mutation*, *dysregulation*, *increased expression*, *decreased expression*, *epigenetic marker*, *therapy resistance*, *prognostic indicator*, *negative prognostic marker*, *positive prognostic marker*, *therapeutic target*, *diagnostic tool*, *genetic susceptibility*.
"""

SYS_PROMPT = """You are a helpful assistant that extracts the list of {"gene":<gene_text>, "disease":<disease_text", "relation":<relation_text>} triplets from the given text. If no triplets are found, please provide an empty list. """
class BioTriplexDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=None, entity_tokens_targets=False,
                 special_tokens=True):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            with open(dataset_config.data_path + "train.txt", "r") as f:
                dataset = f.readlines()
        elif split_name == "val":
            with open(dataset_config.data_path + "val.txt", "r") as f:
                dataset = f.readlines()
        elif split_name == "test":
            with open(dataset_config.data_path + "test.txt", "r") as f:
                dataset = f.readlines()
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        dataset = [json.loads(line) for line in dataset]
        # dataset is split into sentences I want to treat each sentence as a separate example
        new_dataset = []
        for sample in dataset:
            for idx, sentence in enumerate(sample["sentences"]):
                new_sample = {
                    "input": sentence,
                    "output": triplets_to_json(sample["triplets_text"][idx]),
                    "doc_key": sample["doc_key"] + f"_sentence_{idx}",
                    "entities": self.correct_entity_char_index(sample["ner"][idx], sample["sentences"], idx)
                }
                new_dataset.append(new_sample)
        self.data = new_dataset
        if REMOVE_ALL_NEGATIVES:
            self.data = [item for item in self.data if item["output"] != "[]"]
        self.max_words = max_words
        self.tokenizer = tokenizer
        # self.num_truncated_examples = 0
        # self.longest_input = 0
        # self.input_seen = set()
        self.entity_tokens_targets = entity_tokens_targets
        self.special_tokens = special_tokens
        if entity_tokens_targets:
            if special_tokens:
                self.gene_special_token_id = tokenizer.vocab['<|gene token|>']
                self.disease_special_token_id = tokenizer.vocab['<|disease token|>']
                self.relation_special_token_id = tokenizer.vocab['<|relation token|>']
                self.no_entity_special_token_id = tokenizer.vocab['<|no entity token|>']
            else:
                self.gene_special_token_id = tokenizer.vocab['gene']
                self.disease_special_token_id = tokenizer.vocab['condition'] # or 'isease' ?
                self.relation_special_token_id = tokenizer.vocab['relation']
                self.no_entity_special_token_id = tokenizer.vocab['null']


    @staticmethod
    def correct_entity_char_index(entities, sentences, sentence_idx):
        # correct entity character indexes to be relative to the sentence and not the whole text
        offset = sum([len(sentence) for sentence in sentences[:sentence_idx]])
        for entity in entities:
            entity[0] -= offset
            entity[1] -= offset
        return entities

    def get_all_input_prompts(self):
        prompts = {}
        for item in self.data:
            prompt = self.input_to_prompt(item["input"])
            prompts[item["doc_key"]] = prompt
        return prompts

    def input_to_prompt(self, input_text):
        # prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{input_text}\n\n### Response:\n"
        prompt_prefix = f"<|start_header_id|>system<|end_header_id|>{SYS_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>" +\
            f"### Instruction:\n{INSTRUCTION}\n### Input:\n"
        prompt_input = input_text
        prompt_suffix = "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return prompt_prefix, prompt_input, prompt_suffix

    @staticmethod
    def get_entity_indexes(entities, prompt_offsets_mapping):
        genes_indexes = []
        diseases_indexes = []
        relations_indexes = []
        entity_idx = 0
        for idx, (start, end) in enumerate(prompt_offsets_mapping):
            while entity_idx < len(entities):
                entity = entities[entity_idx]
                start_char, end_char = entity[:2]
                if start <= start_char < end or start <= end_char < end or (start_char < start and end_char > end):
                    if entity[2] == "GENE":
                        genes_indexes.append(idx)
                    elif entity[2] == "DISEASE":
                        diseases_indexes.append(idx)
                    elif entity[2] == "RELATION":
                        relations_indexes.append(idx)
                    else:
                        raise ValueError(f"Invalid entity type: {entity[2]}")
                    break
                elif start_char >= end:
                    break
                elif start >= end_char:
                    entity_idx += 1
        assert entity_idx == len(entities), f"Only {entity_idx} out of {len(entities)} entities found in the prompt"
        return genes_indexes, diseases_indexes, relations_indexes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        # self.input_seen.add(index)

        item = self.data[index]

        # prompt = item['input']#f"item['input']\n\n"
        prompt_prefix, prompt_input, prompt_suffix = self.input_to_prompt(item["input"])
        prompt = prompt_prefix + prompt_input + prompt_suffix
        # example = prompt + item["output"]
        example = prompt + "\n### Response:\n" + item["output"]
        if item["output"] != "[]":
            weight = POSITIVE_WEIGHT
        else:
            weight = NEGATIVE_WEIGHT
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        # self.longest_input = max(self.longest_input, example.shape[0])
        if self.max_words is not None:
            raise NotImplementedError("max_words is not implemented")
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]
                # self.num_truncated_examples += 1
        labels = copy.deepcopy(example)
        if self.entity_tokens_targets:
            prompt_prefix = self.tokenizer.encode(prompt_prefix)
            prompt_input = self.tokenizer(prompt_input, add_special_tokens=False, return_offsets_mapping=True)
            prompt_offsets_mapping = prompt_input["offset_mapping"]
            prompt_input = prompt_input["input_ids"]
            prompt_suffix = self.tokenizer.encode(prompt_suffix, add_special_tokens=False)
            labels[:len(prompt_prefix)] = -1
            labels[len(prompt_prefix): len(prompt_prefix) + len(prompt_input)] = self.no_entity_special_token_id
            genes_indexes, diseases_indexes, relations_indexes = self.get_entity_indexes(item["entities"],
                                                                                         prompt_offsets_mapping)
            labels[genes_indexes] = self.gene_special_token_id
            labels[diseases_indexes] = self.disease_special_token_id
            labels[relations_indexes] = self.relation_special_token_id
            labels[len(prompt_prefix) + len(prompt_input): len(prompt_prefix) + len(prompt_input) + len(prompt_suffix)] = -1
        else:
            prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
            labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        # label_mask = label_mask.float()
        # example[example == -100] = self.tokenizer.pad_token_id
        # labels[labels == -100] = self.tokenizer.pad_token_id

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
            "weight": weight,
            # "doc_key": item["doc_key"],
            # "label_mask": label_mask
        }


if __name__ == "__main__":
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|gene token|>",
                                                                "<|disease token|>",
                                                                "<|relation token|>",
                                                                "<|no entity token|>"]})
    from llama_recipes.configs.datasets import biotriplex_dataset
    dataset_config = biotriplex_dataset
    for mode in "val", : #"train", "test":
        dataset = BioTriplexDataset(dataset_config, tokenizer, mode, max_words=None, entity_tokens_targets=True)
        # print number of positive and negative examples (with weight 1 and 0.1 respectively)
        num_positive = 0
        num_negative = 0
        # for i in range(len(dataset)):
        #     if dataset[i]["weight"] == POSITIVE_WEIGHT:
        #         num_positive += 1
        #     else:
        #         num_negative += 1
        # print("MODE:", mode)
        # print(num_positive, num_negative)
        # # print len of longest input
        # max_len = 0
        # for i in range(len(dataset)):
        #     max_len = max(max_len, len(dataset[i]["input_ids"]))
        # print(max_len)
