# ClimatEM

This is the code to run experiments to train and evaluate probabilistic generative models that aim to emulate climate models.


## Installation

### Dependencies
- Python 3.10
- poetry
- [accelerate](https://huggingface.co/docs/accelerate/index)
- wandb

### 1. Clone the repo

``git clone git@github.com:RolnickLab/climatem.git``

### 2. Set up environment
In order to set up the environment, please use venv, running the following commands once the repository is cloned.

Environment creation, first make sure you have a python installation (or `module load python\3.10` on cluster), and run from the climatem directory:

1. `python3 -m venv env_emulator_climatem`

2. `source env_emulator_climatem/bin/activate`

3. `poetry install`
Run `poetry install --with dev` to use formatting tools for code development

If you do not have poetry yet, follow guidelines in the "Install poetry" section here https://github.com/RolnickLab/lab-basic-template
This link points you to additional references for setting up your environment correctly. 

4. Install `pre-commit` hooks using `pre-commit install`.

### 3. Downloading input data

For running the model on real cliamte data, please download monthly climate model data and regrid it to an icosahedral grid using ClimateSet https://github.com/RolnickLab/ClimateSet. 
If you have any problem downloading or formatting the data, please get in touch. 

### 4. Running the model

An example of bash script can be found in scripts/run_single_jsonfile.sh. 
This script assumes that environment `env_climatem` and `climatem` repo are located in `$HOME`. 
You should then update the parameters of the config file `configs/single_param_file.json` and the bash script params at top of the bash script. 
If you use a different json file, set its location using `--config-path path_to_json_params.json` in the bash script. 
For overwriting parameters of the json file, you can also add `--hp train_params.batch_size=256` to the command line (this is an example of setting batch_size, a parameter in the train_params config object, to 256)

Detailed description of parameters can be found in `climatem/config.py`. 
`configs/single_param_file.json` and `configs/single_param_file_savar.json` show an example of parameter files used for climate model and synthetic data respectively. 

To run the particle filter, run the bash script scripts/run_rollout_bf.sh using the same param json file. The rollout parameters can be set in the json file as well.  

#### accelerate
Parallelism is handled by Accelerator https://huggingface.co/docs/accelerate/package_reference/accelerator

Set up accelerate by running 

``accelerate config``

It will prompt a range of questions. You can either answer them yourself or copy paste the following configs into your `$HOME/.cache/huggingface/accelerate/default_config.yaml`

```
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: true
gpu_ids: 0,1,2,3
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
Check configurations by running:

``accelerate env``

Run a quick test to check if it's working:

``accelerate test``

If this runs smoothly you can go ahead.

#### jupyter

In order to run notebooks, you need to install a jupyter kernel. 
To do so, first activate yoru environment and run 

```
python -m pip install jupyter
python -m ipykernel install --user --name=my_env_name
```







