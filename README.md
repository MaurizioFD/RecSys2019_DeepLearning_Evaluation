# DeepLearning RS Evaluation

## Full results and hyperparameters
The full results and corresponding hyperparameters for all DL algorithms are accessible [HERE](FULL_RESULTS.md).

## Installation

Note that this repository requires Python 3.6

First we suggest you create an environment for this project using virtualenv (or another tool like conda)

If you are using virtualenv, first checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment:
```Python
virtualenv -p python3 DLevaluation
source DLevaluation/bin/activate
```

Then install all the requirements and dependencies
```Python
pip install -r requirements.txt
```

In order to compile you must have installed: _gcc_ and _python3 dev_, which can be installed with the following commands:
```Python
sudo apt install gcc 
sudo apt-get install python3-dev
```

Then compile Cython algorithms. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```Python
python run_compile_all_cython.py
```


## Run the experiments

After the installation is complete you can run the experiments.

The DL algorithms are contained in the "Conferences" folder, and further divided in the conference they were published in. The repository has, for each DL algorithm, two folders:
* A folder named "_github" which contains the full original repository, with the minor fixes needed for the code to run.
* A folder named "_our_interface" which contains the python wrappers needed to allow its testing.

All experiments related to a DL algorithm reported in our paper can be executed by running the corresponding script, which is preceeded by _run_ParameterSearch_.
For example, if you want to run the experiments for SpectralCF, you should run this command:
```Python
python run_ParameterSearch_RecSys_18_SpectralCF.py
```

The script will:
* Load the original data split, if available. If not, automatically download the dataset and perform the split with the appropriate methodology. If the dataset cannot be downloaded automatically, a console message will display the link at which the dataset can be manually downloaded and instructions on where the user should save the compressed file.
* Run the bayesian hyperparameter optimization on all baselines, saving the best values found.
* Run the fit and test of the DL algorithm
* Create the latex code of the result tables, as well as plot the data splits, when required. 
* The results can be accessed in the _result_experiments_ folder.
 