# GAN_Project1

Precondition:
Windows users can follow the official microsoft tutorial to install python, git and vscode here:

- ​​https://docs.microsoft.com/en-us/windows/python/beginners
- german: https://docs.microsoft.com/de-de/windows/python/beginners

## Visual Studio Code

This repository is optimized for [Visual Studio Code](https://code.visualstudio.com/) which is a great code editor for many languages like Python and Javascript. The [introduction videos](https://code.visualstudio.com/docs/getstarted/introvideos) explain how to work with VS Code. The [Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial) provides an introduction about common topics like code editing, linting, debugging and testing in Python. There is also a section about [Python virtual environments](https://code.visualstudio.com/docs/python/environments) which you will need in development. There is also a [Data Science](https://code.visualstudio.com/docs/datascience/overview) section showing how to work with Jupyter Notebooks and common Machine Learning libraries.

The `.vscode` directory contains configurations for useful extensions like [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens0) and [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python). When opening the repository, VS Code will open a prompt to install the recommended extensions.

## Development Setup

Open the [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) and run the setup script for your OS (see below). This will install a [Python virtual environment](https://docs.python.org/3/library/venv.html) with all packages specified in `dependencies.txt`.

### Linux and Mac Users

1. run the setup script: `./setup.sh` or `sh setup.sh`
2. activate the python environment: `source .venv/bin/activate`
3. run example code: `python src/hello.py`
4. install new dependency: `pip install sklearn`
5. save current installed dependencies back to dependencies.txt: `pip freeze > dependencies.txt`

### Windows Users

1. run the setup script `.\setup.ps1`
2. activate the python environment: `.\.venv\Scripts\Activate.ps1`
3. run example code: `python src/hello.py`
4. install new dependency: `pip install sklearn`
5. save current installed dependencies back to dependencies.txt: `pip freeze > dependencies.txt`

## ML training
If you want to start a new training please follow this instruction:

0. Delete content in folders `.\data\model`, `.\data\evaluation` and `.\data\images`
   if those folders are not empty the already existing trainingsdata will be used
1. edit training- and evaluation-parameter in file: `.\config.ini`
2. copy the domain-images ('real images') in format 'jpg' to folder `.\data\images\1` and `.\data\images\2` 
3. run script for data-preparation: `python src\dataPreperation.py`
    -> a new pickle-file will be written : `data\evaluation\images_real.pkl`
4. run script `python .\src\training.py` to start the training.
    -> GAN-models an samples by the generator will be saved in folder `.\data\evaluation`
    as pickle-file
   (If the training.py will be started a second time, the saved GAN-model will be used)

## ML evaluation
1. make sure file `.\data\evaluation\generated_images.pkl` is available.
2. By runnning script `.\src\evaluate.py` the sampled images saved as picke-file will be created as jpg-Images in folder `.\data\evaluation`

## Generate images with trained Generator
1. run script  `.\data\test\create.py`

Troubleshooting: 

- If your system does not allow to run powershell scripts, try to set the execution policy: `Set-ExecutionPolicy RemoteSigned`, see https://www.stanleyulili.com/powershell/solution-to-running-scripts-is-disabled-on-this-system-error-on-powershell/
- If you still cannot run the setup.ps1 script, open it and copy all the commands step by step in your terminal and execute each step
