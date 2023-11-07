# MOBO4EIC
a repo for developing MOBO tools for EIC. 

The latest optimization pipeline can be monitored in `weights and biases` [here](https://wandb.ai/phys-meets-ml/AID2E-Closure-1/workspace?workspace=user-karthik18495)

## Installation

The following installation steps are followed

1. Install anaconda/miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)
2. create an empty conda environment `conda create --env ENV_NAME` and activate `conda activate ENV_NAME`
3. After creating, install pip as `conda install pip` and run `pip install pip_requirements.txt`
4. Then continue with `conda install --file conda_requirements.txt`
5. To upload to the existing, project in `weights & biases`, make sure to sign up for an account, and let me know, so I can add teammates. Else, create a new project and modify `temp_secrets.key` to add the relevant API key

## Usage

In order to run an optimization use the following command after loading the relevant `environment` in `conda`

```bash
>> cd MOBO-Closures
>> python wrapper.py -c optimize.config -p True -s secrets.key
```

This should run the optimization. Modify any hyperparameters for optimization in the `optimize.config` file. 

## To do

- Make the framework more modular. Seperate each of the data classes seperately
- Write more detailed documentation using `git-book` or `jupyter-book`
