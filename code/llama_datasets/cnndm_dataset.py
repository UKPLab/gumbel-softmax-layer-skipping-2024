# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/cnn_dailymail

import datasets

from llama_datasets.utils import Concatenator

def get_preprocessed_cnndm(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("cnn_dailymail", split=split, ignore_verifications=True) 


    prompt = (
        f"Summarize this article:\n{{article}}\n---\nSummary:\n{{highlights}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                article=sample["article"],
                highlights=sample["highlights"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
