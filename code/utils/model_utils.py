# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import os
import torch

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import BitsAndBytesConfig

from neuralnets.llama_gumbel import LlamaForCausalLMGumbel

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model

# Loading the model from config to load FSDP checkpoints into that
def load_gumbel_llama(model_name, quantization):
    # First load architecture
    model = LlamaForCausalLMGumbel.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    # Load weights
    model.load_state_dict(torch.load(os.path.join(model_name,"model.pt"),map_location='cuda:0'))
    return model
    
# Loading the model from config to load FSDP checkpoints into that
def load_gumbel_llama_huggingface(model_name, quantization):
    # First load architecture
    model = LlamaForCausalLMGumbel.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model
