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


## Getting Started

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
