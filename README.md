# ClimatEM

This is the code to run experiments to train and evaluate probabilistic generative models that aim to emulate climate models.

In order to set up the environment, please use venv, running the following commands once the repository is cloned.

Environment creation, first make sure you have a python installation, and run from the climatem directory:

python3 -m venv env_emulator_climatem

source activate env_emulator_climatem

pip install -r requirements_env_emulator.txt    

P.S. (this still has a left over installation from the causalpaca github that I have tried to make sure we don't need any more, but I need to check this)

Then follow poetry instructions.

You may also need to create a directory: climatem/Climateset_DATA/

Ok!











### See lab-basic-template readme below:

# README

Set up a minimal codebase to run climate emulation experiments with causal representation learning.

## Description

Repository template that focuses on simplicity and ease of use. 

This is ideal for quick projects or code publication.

The purpose of this template is to help with code quality, structure and 
reproducibility.

This template is not intended to be used as is for libraries or applications that will
be maintained over time, as several things are missing from it, like change logs, 
advanced tools and coding standards (though it can be expanded for such uses).

## Initialization

Please follow these steps:

1. Clone or download the `lab-basic-template` repository (this repository)
2. Either start a new GitHub repository, or select an existing one (the target repository)
3. Copy the files and folders of the `lab-basic-template` repository into your target repository.
   * Do not copy the `.git` folder from the `lab-basic-template`.
   * If you have existing code:
     * Modules (python code that is meant to be _imported_ in other python files) should go into the [src folder](src)
     * Python scripts that only import packages and are meant to be executed on the cluster 
       (including your own modules above) should go into the [scripts folder](scripts)
4. Set the module name by replacing `lab_basic_template` on line #2 of the [pyproject.toml](pyproject.toml) file with 
   your own.
   1. Make sure the name in `snake_case`, like other python modules.
      1. Bad examples : `my-module`, `MyModule`, `My Module`
      2. Good example : `my_module`
   2. Optional steps :
      1. Rename [src folder](src) to your module name
      2. Then remove line #3 in [pyproject.toml](pyproject.toml); `packages = [{include = "*", from="src"}]`
      3. This will allow you to use `from <module_name> import ...` instead of `from src import ...` 
5. Write your name on line #6 in [pyproject.toml](pyproject.toml), and write a short description on line #5
6. Follow the rest of the instructions in this README
7. Remove this section (_Initialization_) from the README of your target repository and modify it's title 
   and description

**Important note**
If you are planning to use this for a new project and expect to use the DRAC cluster 
as well as other clusters/locations, it is recommended to first set up your environment 
on DRAC, as the versions of Python libraries are often a bit behind compared to the Mila 
cluster.

This will make your project more portable and will prevent many dependency management 
problems while working across different clusters.

Installing this module for the first time (see [Installation](#installation)) 
will create the `poetry.lock` file, which will set the different library versions used 
by the project, and therefore help with reproducibility and reduce the classic but 
annoying "but it works on my machine" situation.

However, this `poetry.lock` file can be problematic when using locally compiled python 
wheels, [like is recommended in their documentation](#drac-specific-install-directives).

If working on multiple different clusters, it might be better to add the `poetry.lock` 
file to your `.gitignore`, and manage your dependencies with either explicit versions or 
with [Tilde requirements](https://python-poetry.org/docs/dependency-specification/#tilde-requirements).
 
## Python Version

This project uses Python version 3.11

## Build Tool

This project uses Poetry as a build tool. Using a build tool has the advantage of 
streamlining script use as well as fix path issues related to imports.

## Environment Management

You will need to create a virtual environment for your dependencies.

* [How to create a virtual environment for the Mila cluster](docs/environment_creation_mila.md)
* [How to create an environment for the DRAC cluster](docs/environment_creation_drac.md)
* [How to create a Conda environment](docs/conda_environment_creation.md)
* [Migrating to DRAC from another environment](docs/migrating_to_drac.md)

Do note that Conda is not available on the DRAC cluster, and there are some extra steps
to use Conda on the Mila cluster compared to a workstation.

### Poetry

Once an environment has been set up and documented, Poetry needs to be installed if it 
is not already available on your system.

For more information on [Poetry](https://python-poetry.org/docs/), 
consult the official documentation and the [Contribution guidelines](CONTRIBUTING.md).

*Disclaimer for those that already know a lot about Poetry...*

Yes, `Poetry` can manage environments directly, and there are a lot of other more advanced 
uses that are not explored in this repository. This is done on purpose, as an introduction 
to this tool in a context that is familiar for most users (i.e. creating virtual environments
with venv/virtualenv/conda). If you are comfortable with `Poetry` and especially its use 
on compute clusters, feel free to disregard the recommendations below. Just don't forget 
to document its use for the project!

#### Virtual Environment (venv or virtualenv)

If using a virtual environment, `Poetry` should be installed through `pipx`

In your activate virtual environment:

```
pip install pipx
pipx install poetry
```

This will install `Poetry` separately from this repository's dependencies, and make it 
available for other projects too! If ever you want to remove `Poetry`, the following 
command can be used to uninstall it : `pipx uninstall poetry`

Unless you want to keep `pipx` in your activated virtual environment, it can be removed 
using the `pip uninstall pipx`, which will not affect the `Poetry` installation.

#### Conda Environment

If using a Conda environment, Poetry can be installed in your activated project environment:

```
conda install poetry
```

See [Creating a Conda Environment](docs/conda_environment_creation.md) for more 
information on how to get started with Conda.

## Installation

Once the virtual environment and `Poetry` is installed, follow these steps:

1. Make sure your virtual environment is active
2. Install the module and its dependencies using the following command:
    * `poetry install`
    * Alternatively, you can also install using `pip install -e .`, which will install 
      your module, [configured scripts](https://python-poetry.org/docs/pyproject#scripts) 
      and dependencies, but without creating a `poetry.lock` file.

### DRAC Specific Install Directives

This is in accordance with the official [DRAC documentation](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment), 
adapted for use with `Poetry`.

It is common practice, though not enforced, to re-create your Python environment inside 
your job. Since most nodes on the DRAC cluster do not have access to the internet, the 
dependencies are therefore installed from pre-built wheels. If this does not apply to 
you, or if you work on Cedar, which nodes do have access to the internet, you can 
disregard the following.

To install the dependencies from these wheels, use the following steps:

1. Create your environment like specified in [environment_creation_drac.md](docs/environment_creation_drac.md)
2. Instead of using `poetry install`, use `pip install -e . --no-index`
   * This will install the module, [configured scripts](https://python-poetry.org/docs/pyproject#scripts) 
     as well as the dependencies. However, `pip` can only install main dependencies and 
     will not be able to install Poetry defined groups

## Development

If you want to contribute to this repository, the development dependencies will also need to added.

1. Install `pre-commit` and other dev dependencies using the following command:
   * `poetry install --with dev`
     * `pre-commit` is used for code quality and code analysis
2. Optional Pylint check
   * While not enforced by the pre-commit tool, running Pylint on your code can help
     with code quality, readability and even catch errors or bad coding practices.
   * To run this tool, 
3. Python library dependencies
   * To keep things simple, it is recommended to store all new dependencies as main dependencies. 
4. Read and follow the [Contributing guidelines](CONTRIBUTING.md)


