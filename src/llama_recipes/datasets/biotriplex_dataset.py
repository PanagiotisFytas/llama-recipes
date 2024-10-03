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


class BioTriplexDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split_name, max_words=512):
        #self.data = json.load(open(dataset_config.data_path))

        if split_name == "train":
            with open(dataset_config.data_path + "train.txt", "r") as f:
                dataset = f.readlines()
        else:
            with open(dataset_config.data_path + "val.txt", "r") as f:
                dataset = f.readlines()
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
        self.data = dataset

        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        item = self.data[index]

        #prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
        prompt = item['input']#f"item['input']\n\n"

        example = prompt + item["output"]
#        print(example)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


if __name__ == "__main__":
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    dataset_config = "data/biotriplex/"
    dataset = BioTriplexDataset(dataset_config, tokenizer, "train")
    print(dataset[0])
    print(tokenizer.decode(dataset[0]["input_ids"]))
    print(tokenizer.decode(dataset[0]["labels"]))
    print(dataset[0]["attention_mask"])
    print(dataset[0]["input_ids"].shape)
    print(dataset[0]["labels"].shape)
    print(dataset[0]["attention_mask"].shape)
    print(dataset[0]["input_ids"].dtype)
    print(dataset[0]["labels"].dtype)
    print(dataset[0]["attention_mask"].dtype)