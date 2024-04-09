# Rebuilding ROME : Resolving Model Collapse during Model Editing

This repo builds on [Rebuilding ROME](https://github.com/scalable-model-editing/rebuilding-rome) and the [original ROME codebase](https://github.com/kmeng01/ROME).

## Changes to the update equation

We focus on the way the MLP keys ($`k`$ such that $`Wk=v`$) are computed. See the `rome/compute_u.py` and `rome/compute_v.py` scripts for details.
The derived ROME update equation is

$$\hat{W}=W+\Lambda_* \left(C^{-1} k_*\right)^T$$

where

$$\Lambda_* =\frac{v_* - Wk_* }{ (C^{-1} k_* )^T }$$

$$k_* =\frac{1}{N} \sum_{j=1}^N k\left(x_j+s\right).$$

$$k(x)= \sigma \left( W_{f c}^{\left( l^* \right)} \gamma\left( a_{\[x\], i}^{\left( l^* \right)} + h_{\[x\], i}^{\left( l^* -1 \right)} \right)\right)$$

Note that the optimization step to compute $`v_*`$ is based on $`k_*`$. The original ROME implementation, however, computes

$$\hat{W}=W+\Lambda\left(C^{-1} k_*\right)^T$$

where

$$\Lambda=\frac{v_* -Wk}{(C^{-1} k)^T}$$
$$k = k(s)$$

We find that the latter leads to rapid degradation in model performance in a sequential editing setting, and prone to particular edits known as *disabling edits* that render the model unusable post-update. Our experiments focus on unifying the computation of the keys in the update equation, and we study the use of $`k`$ and $`k_*`$.

## Installation

We recommend using Docker to set up a clean dev environment.

`docker compose up -d --build`

To download the datasets used for evaluation, install Git LFS if needed:

```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
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
