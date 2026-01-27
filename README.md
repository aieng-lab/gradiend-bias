# GRADIEND: Feature Learning within neural networks exemplified through biases
> Jonathan Drechsel, Steffen Herbold
[![arXiv](https://img.shields.io/badge/arXiv-2502.01406-blue.svg)](https://arxiv.org/abs/2502.01406)

This repository contains the official source code for the training and evaluation of [GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models](https://arxiv.org/abs/2502.01406).
Further evaluations of this study can be reproduced using our [expanded version of bias-bench](https://github.com/aieng-lab/bias-bench).

## Quick Links
- [GRADIEND Paper](https://arxiv.org/abs/2502.01406)
- GRADIEND Training and Evaluation Datasets (Hugging Face):
  - [GENTER](https://huggingface.co/datasets/aieng-lab/genter) (Gender Data)
  - [Deprecated: GENEUTRAL](https://huggingface.co/datasets/aieng-lab/geneutral)
  - [BIASNEUTRAL](https://huggingface.co/datasets/aieng-lab/biasneutral)
  - [GENTYPES](https://huggingface.co/datasets/aieng-lab/gentypes)
  - [NAMEXACT](https://huggingface.co/datasets/aieng-lab/namexact)
  - [NAMEXTEND](https://huggingface.co/datasets/aieng-lab/namextend)
  - [Race Data](https://huggingface.co/datasets/aieng-lab/gradiend_race_data)
  - [Religion Data](https://huggingface.co/datasets/aieng-lab/gradiend_religion_data)
- GRADIEND Gender Debiased Models (Hugging Face):
  - [`aieng-lab/bert-base-cased-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/bert-base-cased-gradiend-gender-debiased)
  - [`aieng-lab/bert-large-cased-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/bert-large-cased-gradiend-gender-debiased)
  - [`aieng-lab/distilbert-base-cased-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/distilbert-base-cased-gradiend-gender-debiased)
  - [`aieng-lab/roberta-large-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/roberta-large-gradiend-gender-debiased)
  - [`aieng-lab/gpt2-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/gpt2-gradiend-gender-debiased)
  - [`aieng-lab/Llama-3.2-3B-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/Llama-3.2-3B-gradiend-gender-debiased)
  - [`aieng-lab/Llama-3.2-3B-Instruct-gradiend-gender-debiased`](https://huggingface.co/aieng-lab/Llama-3.2-3B-Instruct-gradiend-gender-debiased)
- Relevant Repositories:
  - [`aieng-lab/bias-bench`](https://github.com/aieng-lab/bias-bench) for evaluation
  - [`aieng-lab/lm-evaluation-harness`](https://github.com/aieng-lab/lm-evaluation-harness) for GLUE zero-shot evaluation


## Install
```bash
git clone https://github.com/aieng-lab/gradiend.git
cd model_with_gradiend
conda env create --file environment.yml
conda activate model_with_gradiend
```

Download [Gendered Words](https://github.com/ecmonsen/gendered_words) and copy the file into the `data/` directory of this repository.

Download [bias-attribute-words](https://github.com/aieng-lab/bias-bench/blob/main/data/bias_attribute_words.json) and copy the file into the `data/` directory of this repository.

Optional: Install [`aieng-lab/bias-bench`](https://github.com/aieng-lab/bias-bench) for further evaluations and comparison to other debiasing techniques.

In order to use Llama-based models, you must first accept the Llama 3.2 Community License Agreement (see e.g., [here](https://huggingface.co/meta-llama/Llama-3.2-3B)). Further, you need to export a variable `HF_TOKEN` with a HF access token associated to your HF account (alternatively, but not recommended, you could insert your HF token in `gradiend/model.py#HF_TOKEN`).
Note that the GRADIEND training with default configuration requires three NVIDIA A100 GPUs with 80GB each (all other tested models run on a single A100). The implementation automatically distributes the weights across the available GPUs for Llama.

## Overview

Package | Description
--------|------------
`gradiend.model` | GRADIEND model implementation
`gradiend.data` | Data util functions
`gradiend.training` | Training of GRADIEND
`gradiend.evaluation` | Evaluation of GRADIEND
`gradiend.export` | Export functions for results, e.g., printing LaTeX tables and plotting images
`gradiend.setups` | Predefined setups for training, data, and evaluation, e.g., `GenderEnSetup()`



> **__NOTE:__** All python files of this repository should be called from the root directory of the project to ensure that the correct (relative) paths are used (e.g., `python gradiend/training/gradiend_training.py`).

See `demo.ipynb` for a quick overview of the GRADIEND model and the evaluation process.
(TODO this might be outdated!?!)


### Setups

As the feature learnt by GRADIEND depends only on the data and task used during training (e.g., predicting gender related tokens results in gender feature), we provide several pre-defined *setups* which combine the data, training, and evaluation for a specific task.


#### GenderEnSetup

This setup learns a binary gender feature in English based on templated texts associating gendered singular third-person pronouns (he/she) to first names.
```
[NAME] plays soccer. [PRONOUN] is very good at scoring goals.
```

- [Optional:] Generate the data by running `gradiend.setups.gender.en.filtering.py`. This is optional as the generated datasets are also provided via Hugging Face.
- Run the training with `gradiend.setups.gender.en.training.py`. In the default configuration, this script will train 3 GRADIEND models for 7 base models (`bert-base-cased`, `bert-large-cased`, `distilbert-base-cased`, `roberta-large`, `gpt2`, `Llama-3.2-3B`, and `Llama-3.2-3B-Instruct`), selecting the best GRADIEND model at the end, followed by analysis of the encoder and decoder.
- During training, the encoding gets normalized such that the female gradients are encoded positively (mostly).


#### Race and Religion Setups

Race and religion are considered with three classes each (*Asian/Black/White* and *Christian/Muslim/Jewish*).
Hence, we provide a setup for each pair in `gradiend.setups.race_religion.training`, e.g, `WhiteBlackSetup` and `ChristianMuslimSetup`.
Training for all setups and all considered base models can be done via `gradiend.setups.race_religion.training.py`.
Encoder statistics are computed with `gradiend.setups.race_religion.analyze.py` and can be plotted with `gradiend.setups.race_religion.encoder_stats.py`. 


### Training

The training of the Gender GRADIEND models is done by running the `gradiend.training.gradiend_training` script.
Intermediate results are saved in `results/experiments/gradiend`, and the final models are saved in `results/models`.
The `gradiend_training` script relies on `gradiend.training.trainer.train()` function trains a single GRADIEND model and provides many hyperparameters, see [here](gradiend/training/trainer.py) for details.

### Evaluation

#### Analysis of Decoder and Generation of (De-)Biased Models

`gradiend.evaluation.analyze_decoder.default_evaluation()` evaluates the decoder of a trained GRADIEND model by generating debiased models for different learning rates and gender factors.
The evaluation results are cached per learning rate and gender factor (`results/cache/decoder`), and plots are shown visualizing the results.

For gender, the best debiased, male-biased, and female-biased models according to this evaluation can be generated by executing the `gradiend.evaluation.select_models` script, which saves these models into `results/changed_models`. The models are names `[base model]-[type]`, with type being `N` for the debiased model, `F` for the female model, and `M` for the male model.

Some basic evaluations of these debiased models can be done by calling:
- `gradiend.analyze_decoder.evaluate_all_gender_predictions()` and `gradiend.export.gender_predictions.py` for an overfitting analysis
- `gradiend.export.example_predictions.py` to generate example predictions

For race and religion, call `setups.race_religion.analyze.py` and `setups.race_religion.model_selection.py`.


### Evaluation of (De-)Biased Models
See [bias-bench](https://github.com/aieng-lab/bias-bench) for a comparison of the (de-)biased models generated with GRADIEND to other debiasing techniques.

### Export
The export package contains functions to export the results of the evaluations, e.g., to print LaTeX tables or to plot images.

Script | Description
-------|------------
`dataset_stats` | prints the statistics of the datasets used in the paper
`encoder_plot` | Plots a violin plot regarding the distribution of encoded values of the encoder analysis
`changed_model_selection` | Generates a table with the statistics of the selected gender (de-) biased models
`gender_predictions` | Plots predicted female and male probabilities for simple masking task to evaluate overfitting
`example_predictions` | Generates example predictions for the selected gender (de-) biased as a LaTeX table
`setups.race_religion.model_selection` | Generates plots and tables regarding the model selection
`setups.race_religion.encoder_stats` | Generates plots and tables regarding the encoder analysis


> **__NOTE:__** To enable LaTeX plotting with your desired font, you need to adjust the `init_matplotlib()` function default arguments in the gradiend.util.py` file.

## Dataset Generation

Although the experiments mentioned above are based on data published on Hugging Face by now, we also provide the code to 
generate the datasets used in the paper.

### Required Datasets
If you want to re-create the datasets generated in the paper, you first need to download the following datasets:

Dataset | Download Link | Notes                                            | Download Directory
--------|---------------|--------------------------------------------------|-------------------
Gender by Name | [Download](https://doi.org/10.24432/C55G7X) | Required for the generation of the name datasets | `data/`

### Dataset Generation
The following scripts will generate the datasets used in the paper:

Dataset | Generation Script
--------|------------------
GENTER  | `gradiend.setups.gender.en.filtering.generate_genter()`
BIASNEUTRAL | `gradiend.setups.gender.en.filtering.generate_biasneutral()`
NAMEXACT | `gradiend.data.generate_namexact()`
NAMEXTEND | `gradiend.data.generate_namextend()`
Race Data | `gradiend.setups.race_religion.data.generate_race_data()`
Religion Data | `gradiend.setups.race_religion.data.generate_race_data()`

## Citation
```
@misc{drechsel2025gradiendfeaturelearning,
      title={{GRADIEND}: Feature Learning within Neural Networks Exemplified through Biases}, 
      author={Jonathan Drechsel and Steffen Herbold},
      year={2025},
      eprint={2502.01406},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.01406}, 
}
```
