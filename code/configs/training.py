# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="llama-7b"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=1
    gradient_accumulation_steps: int=1
    num_epochs: int=3 
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "model_output"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/storage/ukp/work/lee/intel_ukp_llm/intel_ukp_llm/llama-13b" # will be used if using FSDP
    dist_checkpoint_folder: str="checkpoints" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    gumbel: bool = False # Enable Gumbel softmax for sampling
    gumbel_temperature: float = 1.0 # Gumbel softmax temperature
    gumbel_hard: bool = False # Use hard Gumbel softmax
    gumbel_noskip_low: int = 2 # Layer to start skipping (low)
    gumbel_noskip_high: int = 32 # Layer to stop skipping (high)
    debugging: bool = False # Enable debugging mode
    debugging_host: str = "localhost" # Debugging host
    debugging_port: int = 5678 # Debugging port
    gumbel_target: float = 0.8 # Percent of layers that should be used (for calculating gumbel loss)
    gumbel_loss_multiplier: float = 50.0 # Simple multiplier for gumbel loss
    gumbel_loss_alpha: float = 0.8 # initial weighting factor for the gumbel loss
    gumbel_loss_beta: float = 0.0005 # controls the rate at which the weighting factor decreases
    use_token_max: bool = False # Use max function over token instead of token mean
    use_only_last_token: bool = False # Use only last token for classification
    use_only_past_key_values: bool = False # Use only past key values for classification
    share_layer: bool = False # Share one gumbel layer across all layers
    gumbel_use_simple_classifier: bool = False # Use simple classifier instead of gumbel
    gumbel_num_hidden_layers: int = 1 # Number of hidden layers
    gradient_clipping_value: float = 1.0 # gradient Clipping value

    
    
