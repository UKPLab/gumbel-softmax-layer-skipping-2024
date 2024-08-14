<p  align="center">
  <img src='logo.png' width='200'>
</p>

# gumbel_softmax_layer_skipping_2024
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/gumbel-softmax-layer-skipping-2024/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/gumbel-softmax-layer-skipping-2024/actions/workflows/main.yml)

This repository contains the implementation of a trainable layer skipping mechanism using the Gumbel Softmax function. The code is tailored for Llama2. 

Contact person: [Ji-Ung Lee](mailto:bungobang@yahoo.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Project structure

* `configs` &mdash; contains five different configuration files:
    1. `datasets.py` available datasets, names, and splits
    2. `fsdp.py` settings for training on multiple GPUs and concerning quantization
    3. `inference.py` inference configuration; e.g., max_new_tokens, temperature, etc.
    4. `peft.py` settings for parameter efficient training methods (e.g., LoRA)
    5. `training.py` settings for training, e.g., learning rate, batch size, etc.
* `llama_datasets` &mdash; Data loading and preparation routines
* `model_checkpointing` &mdash; (self explanatory)
* `neuralnets` &mdash; Implementation of the gumbel softmax for Llama2
* `utils` &mdash; various utilities for training, saving/loading models, etc.
* `policies` &mdash; Utilities for FSDP (do not touch)
* `results` &mdash; Result folder (needs to be created)

## Getting Started

To run the experiments, first create a respective virtual env (using e.g., conda):

    conda create --name=<envname> python=3.9
    conda activate <envname>
    pip install -r requirements.txt

We run all our experiments with python 3.9.

## Usage

### Training

To finetune a base model, you can use torchrun with `finetune_llama2.py`

    torchrun --standalone \
        --nnodes 1 --nproc_per_node 2 finetune_llama2.py \  # Use 2 GPUs
        --batch_size_training 1 \
        --model_name "<path-to-model>" \
        --pure_bf16 \
        --num_epochs 3 \
        --output_dir "model_output/llama-7b-trained-gumbel" \
        --gumbel 1  \
        --dataset "samsum_dataset"

Detailed description of all parameters are provided in `configs/training.py`. 

### Inference 

To perform innference, you can use torchrun with `evaluate_llama2.py`.

    torchrun --standalone \
        --nnodes 1 --nproc_per_node 1 evaluate_llama2.py \
        --model_name "model_output/llama-7b-trained-gumbel" \
        --use_gumbel 1 \
        --output_dir "results" 

This will also measure the overall time taken for inference (normalized by the number of generated tokens) and keep track of the layers that were activated for the Gumbel Softmax (written into `activations.json`). The scores and generated text will be written into `.json` and `.tsv` files.

## Cite

This work has no accompanying paper. However you may cite the preliminary work on adaptable adapters that served as a basis for this work.

```
@inproceedings{moosavi-etal-2022-adaptable,
    title = "Adaptable Adapters",
    author = "Moosavi, Nafise  and
      Delfosse, Quentin  and
      Kersting, Kristian  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.274",
    pages = "3742--3753",
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
