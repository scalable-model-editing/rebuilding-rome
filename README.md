# REBUILDING ROME : Resolving Model Collapse during Model Editing


## Installation
We work off of the [MEMIT](https://github.com/kmeng01/memit) codebase, so we'll reference the same installation procedures here: 
"We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`."


## Running the experiments
To evaluate r-rome, run the following command:

```python
python experiments/evaluate_r-rome.py \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --ds_name=cf \
    --sequential=False
```

The above script can also be used to do sequential editing on the model using r-rome. To do so, turn on the --sequential flag to TRUE. With sequential editing, the edited model is evaluated for downstream task performance on 4 GLUE datasets after every 20 edits. The interval can be changed within the code-base. 


The original ROME algorithm can be evaluated using the following script:

```python
python experiments/evaluate.py \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json \
    --ds_name=cf \
    --sequential=False
```

Similarly as above, we've added a sequential flag to evaluate the original ROME algorithm for sequential editing. 


**Before any experiment is run**, there might be need to update ```sys.path.append('/path/to/rebuilding-rome')``` to the path of the parent directory. 



## The Main Update Equation