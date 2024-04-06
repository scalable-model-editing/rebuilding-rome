# Rebuilding ROME : Resolving Model Collapse during Model Editing

This codebase builds on [Rebuilding ROME](https://github.com/scalable-model-editing/rebuilding-rome) and the [original ROME codebase](https://github.com/kmeng01/ROME).

## Changes to the update equation

TODO include math

## Installation

We recommend using Docker to set up a clean dev environment.

`docker compose up -d --build`

To download the datasets used for evaluation, install Git LFS:

```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
sudo apt-get install -y git-lfs`
git lfs pull
```

## Running the experiments

The script supports sequential editing with the `--sequential` flag. With sequential editing, the edited model is evaluated for downstream task performance on 4 GLUE datasets after every 20 edits. The interval can be changed within the code-base.

You can evaluate either GPT2-XL or GPTJ-6B using the appropriate hyperparameter file to configure how the update equation is computed.

```python
python experiments/evaluate.py \
    --model_name=${MODEL_NAME} \
    --hparams_fname=${HPARAM_FILE_NAME} \
    --ds_name=cf \
    --sequential
```
