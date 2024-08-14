<p  align="center">
  <img src='logo.png' width='200'>
</p>

# gumbel_softmax_layer_skipping_2024
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/gumbel-softmax-layer-skipping-2024/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/gumbel-softmax-layer-skipping-2024/actions/workflows/main.yml)

This is the official template for new Python projects at UKP Lab. It was adapted for the needs of UKP Lab from the excellent [python-project-template](https://github.com/rochacbruno/python-project-template/) by [rochacbruno](https://github.com/rochacbruno).

> **Abstract:** This repository contains the implementation of a trainable layer skipping mechanism using the Gumbel Softmax function. The code is tailored for Llama2. 

Contact persons: [Ji-Ung Lee](mailto:bungobang@yahoo.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Getting Started

> **DO NOT CLONE OR FORK**

If you want to set up this template:

1. Request a repository on UKP Lab's GitHub by following the standard procedure on the wiki. It will install the template directly. Alternatively, set it up in your personal GitHub account by clicking **[Use this template](https://github.com/rochacbruno/python-project-template/generate)**.
2. Wait until the first run of CI finishes. Github Actions will commit to your new repo with a "✅ Ready to clone and code" message.
3. Delete optional files: 
    - If you don't need automatic documentation generation, you can delete folder `docs`, file `.github\workflows\docs.yml` and `mkdocs.yml`
    - If you don't want automatic testing, you can delete folder `tests` and file `.github\workflows\tests.yml`
4. Prepare a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install .
pip install -r requirements-dev.txt # Only needed for development
```
5. Adapt anything else (for example this file) to your project. 

6. Read the file [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md)  for more information about development.

## Usage

### Using the classes

To import classes/methods of `gumbel_softmax_layer_skipping_2024` from inside the package itself you can use relative imports: 

```py
from .base import BaseClass # Notice how I omit the package name

BaseClass().something()
```

To import classes/methods from outside the package (e.g. when you want to use the package in some other project) you can instead refer to the package name:

```py
from gumbel_softmax_layer_skipping_2024 import BaseClass # Notice how I omit the file name
from gumbel_softmax_layer_skipping_2024.subpackage import SubPackageClass # Here it's necessary because it's a subpackage

BaseClass().something()
SubPackageClass().something()
```

### Using scripts

This is how you can use `gumbel_softmax_layer_skipping_2024` from command line:

```bash
$ python -m gumbel_softmax_layer_skipping_2024
```

### Expected results

After running the experiments, you should expect the following results:

(Feel free to describe your expected results here...)

### Parameter description

* `x, --xxxx`: This parameter does something nice

* ...

* `z, --zzzz`: This parameter does something even nicer

## Development

Read the FAQs in [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md) to learn more about how this template works and where you should put your classes & methods. Make sure you've correctly installed `requirements-dev.txt` dependencies

## Cite

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
