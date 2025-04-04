# GIT-CXR
GIT-CXR: End-to-End Transformer for Chest X-Ray Report Generation

This repository contains the code used for training the models described in [GIT-CXR: End-to-End Transformer for Chest X-Ray Report Generation](https://arxiv.org/pdf/2501.02598).


## Data Preparation

1. Downlowd text files from [https://physionet.org/content/mimic-cxr/2.0.0/](https://physionet.org/content/mimic-cxr/2.0.0/), use the [https://github.com/MIT-LCP/mimic-cxr](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt) repo to create the section files and place them at `<DATA_PATH>/csvs_orig`.
1. Download files from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/), place the csv files at `<DATA_PATH>/csvs_jpg` and the images (optionally scaled to have the smaller side at least 224) at `<DATA_PATH>/files_224`.
1. Download CheXbert (code + checkpoints) from [https://github.com/stanfordmlgroup/CheXbert](https://github.com/stanfordmlgroup/CheXbert). It will be used for evaluation (clinical accuracy metrics F1); however, this is a distinct step done after training the models.

## How to use the code

Some examples on how to train and evaluate our models are provided in the folder `training_scripts_samples`.

## Citations

```
@article{sirbu2025git,
  title={GIT-CXR: End-to-End Transformer for Chest X-Ray Report Generation},
  author={S{\^\i}rbu, Iustin and S{\^\i}rbu, Iulia-Renata and Bogojeska, Jasmina and Rebedea, Traian},
  journal={arXiv preprint arXiv:2501.02598},
  year={2025}
}
```