# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import os
import json
import time
import csv 
import torch
from transformers import LlamaTokenizer, LlamaTokenizerFast, AutoTokenizer

# Evaluation via huggingface rouge:  https://huggingface.co/spaces/evaluate-metric/rouge 
import evaluate
import datasets

from utils.safety_utils import get_safety_checker
from utils.model_utils import load_model, load_peft_model, load_gumbel_llama

from configs import inference_config
from utils.config_utils import update_config

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def main(**kwargs):
    update_config(inference_config, **kwargs)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(inference_config.seed)
    torch.manual_seed(inference_config.seed)
    
    # Load samsum dataset
    dataset = datasets.load_dataset("samsum", split="test")
    
    print(f"Evaluating on {len(dataset)} instances.")
    
    
    # Load model 
    if inference_config.use_gumbel:
        model = load_gumbel_llama(inference_config.model_name, inference_config.quantization)
    else:
        model = load_model(inference_config.model_name, inference_config.quantization)
        
    if inference_config.peft_model:
        model = load_peft_model(model, inference_config.peft_model)
    
    model.to(torch_device)
    model.bfloat16()
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name)

    # Add hooks for gumbel
    if inference_config.use_gumbel:
        # a dict to store the activations
        gumbel_activation = {}
        def hook_fn(layer, input, output):
            gumbel_activation[layer] = output

        for layer_idx, layer in enumerate(model.model.gumbel_layer_selection):
            model.model.gumbel_layer_selection[layer_idx].register_forward_hook(hook_fn)
        activations=[]

    predictions_generated = []
    tok_len_generated = []
    references = []

    dsample = dataset['dialogue']
    dsummary = dataset['summary']

    start = time.perf_counter()
    with torch.no_grad():
        for sample, summary in zip(dsample,dsummary):
            batch = tokenizer(f"Summarize this dialog:\n{sample}\n---\nSummary:\n", padding='max_length', truncation=True, max_length=inference_config.max_padding_length, return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            references.append(summary)
            outputs = model.generate(
                **batch,
                max_new_tokens=inference_config.max_new_tokens,
                do_sample=inference_config.do_sample,
                top_p=inference_config.top_p,
                temperature=inference_config.temperature,
                min_length=inference_config.min_length,
                use_cache=inference_config.use_cache,
                top_k=inference_config.top_k,
                repetition_penalty=inference_config.repetition_penalty,
                length_penalty=inference_config.length_penalty,
            )
            new_tokens = outputs[0][batch["input_ids"].shape[-1]:]
            tok_len_generated.append(len(new_tokens))
            predictions_generated.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
            if inference_config.use_gumbel:
                # Track activated layers
                layer_activations = {i:{} for i in range(len(outputs))}
                for layer_idx, res in enumerate(gumbel_activation.values()):
                    for idx, activation in enumerate(res):
                        layer_activations[idx][layer_idx] = torch.argmax(activation,dim=1).tolist()
                activations.append(layer_activations)
                
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    
    e2e_inference_time_norm = e2e_inference_time / sum(tok_len_generated)
    print(f"the inference time normalized by number of tokens is {e2e_inference_time_norm} ms")
                
    # Compute scores
    rouge = evaluate.load('rouge')
    results_generated = rouge.compute(predictions=predictions_generated, references=references)
    results_generated["time-total"] = e2e_inference_time
    results_generated["time-per-token"] = e2e_inference_time_norm


    print("Results: ",results_generated)
    
    results_file = f"{inference_config.model_name.split('/')[-1]}_samsum"
    
    if inference_config.use_gumbel:
        results_file += "_gumbel-True"
    else:
        results_file += "_gumbel-False"
    
        
    with open(os.path.join(inference_config.output_dir,f"{results_file}.json"),'w') as f:
        json.dump(results_generated,f)
    
    if inference_config.use_gumbel:
        with open(os.path.join(inference_config.output_dir,f"{results_file}_activations.json"),'w') as f:
            json.dump(activations,f)
        
    with open(os.path.join(inference_config.output_dir,f"{results_file}.tsv"),'w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"')
        for p,r in zip(predictions_generated, references):
            writer.writerow([p,r])
            
            
if __name__ == "__main__":
    fire.Fire(main)
