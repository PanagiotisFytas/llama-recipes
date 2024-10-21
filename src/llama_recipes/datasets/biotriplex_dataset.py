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
   - **Relation**: The type of relationship between the gene and the human disease. These relations of interest are *pathological role*, *causative activation*, *causative inhibition*, *causative mutation*, *modulator decrease disease*, *modulator increase disease*, *biomarker*, *associated mutation*, *dysregulation*, *increased expression*, *decreased expression*, *epigenetic marker*, *therapy resistance*, *prognostic indicator*, *negative prognostic marker*, *positive prognostic marker*, *therapeutic target*, *diagnostic tool*, *genetic susceptibility*.
"""

class BioTriplexDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=None):
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
                    "doc_key": sample["doc_key"] + f"_sentence_{idx}"
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

    def get_all_input_prompts(self):
        prompts = {}
        for item in self.data:
            prompt = self.input_to_prompt(item["input"])
            prompts[item["doc_key"]] = prompt
        return prompts

    @staticmethod
    def input_to_prompt(input_text):
        prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        # self.input_seen.add(index)

        item = self.data[index]

        prompt = self.input_to_prompt(item["input"])
        # prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
        if item["output"] != "[]":
            weight = POSITIVE_WEIGHT
        else:
            weight = NEGATIVE_WEIGHT
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        # self.longest_input = max(self.longest_input, example.shape[0])
        if self.max_words is not None:
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]
                # self.num_truncated_examples += 1
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
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
    from llama_recipes.configs.datasets import biotriplex_dataset
    dataset_config = biotriplex_dataset
    for mode in "train", "val", "test":
        dataset = BioTriplexDataset(dataset_config, tokenizer, "train", max_words=None)
        # print number of positive and negative examples (with weight 1 and 0.1 respectively)
        num_positive = 0
        num_negative = 0
        for i in range(len(dataset)):
            if dataset[i]["weight"] == POSITIVE_WEIGHT:
                num_positive += 1
            else:
                num_negative += 1
        print("MODE:", mode)
        print(num_positive, num_negative)
        # print len of longest input
        max_len = 0
        for i in range(len(dataset)):
            max_len = max(max_len, len(dataset[i]["input_ids"]))
        print(max_len)
        # print(dataset[0]["input_ids"].shape)
        # print(dataset[0]["labels"].shape)
        # print(dataset[0]["attention_mask"].shape)
        # print(tokenizer.decode(dataset[0]["input_ids"]))
        # print(tokenizer.decode(dataset[0]["labels"]))
        # print(dataset[0]["attention_mask"])
        # print(dataset[0]["input_ids"].dtype)
        # print(dataset[0]["labels"].dtype)
        # print(dataset[0]["attention_mask"].dtype)