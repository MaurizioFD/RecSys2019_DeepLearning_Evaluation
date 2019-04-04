# DeepLearning RS Evaluation

## Full results and hyperparameters
The full results and corresponding hyperparameters for all DL algorithms are accessible [HERE](FULL_RESULTS.md).
For information on the requirements and how to install this repository, see the following [Installation](#Installation) section.


## Code organization
This repository is organized in several subfolders.

#### Deep Learning Algorithms
The Deep Learning algorithms are all contained in the _Conferences_ folder and further divided in the conferences they were published in.
For each DL algorithm the repository contains two subfolders:
* A folder named "_github" which contains the full original repository, with the minor fixes needed for the code to run.
* A folder named "_our_interface" which contains the python wrappers needed to allow its testing in our framework. The main class for that algorithm has the "Wrapper" suffix in its name. This folder also contain the functions needed to read and split the data in the appropriate way.

Note that in some cases the original repository contained also the data split used by the original authors, that is included as well.

#### Baseline algorithms
Other folders like KNN and GraphBased contain all the baseline algorithms we have used in our experiments.

#### Evaluation
The folder _Base.Evaluation_ contains the two evaluator objects (_EvaluatorHoldout_, _EvaluatorNegativeSample_) which compute all the metrics we report.

#### Data
The data to be used for each experiments is gathered from specific _DataReader_ objects withing each DL algoritm's folder. 
Those will load the original data split, if available. If not, automatically download the dataset and perform the split with the appropriate methodology. If the dataset cannot be downloaded automatically, a console message will display the link at which the dataset can be manually downloaded and instructions on where the user should save the compressed file.

The folder _Data_manager_ contains a number of _DataReader_ objects each associated to a specific dataset, which are used to read datasets for which we did not have the original split. 

Whenever a new dataset is parsed, the preprocessed data is saved in a new folder called _Data_manager_split_datasets_, which contains a subfolder for each dataset and then a subfolder for each conference.

#### Hyperparameter optimization
Folder _ParameterTuning_ contains all the code required to tune the hyperparameters of the baselines. The script _run_parameter_search_ contains the fixed hyperparameters search space used in all our experiments.
The object _SearchBayesianSkopt_ does the hyperparameter optimization for a given recommender instance and hyperparameter space, saving the explored configuration and corresponding recommendation quality. 




## Run the experiments

See see the following [Installation](#Installation) section for information on how to install this repository.
After the installation is complete you can run the experiments.


All experiments related to a DL algorithm reported in our paper can be executed by running the corresponding script, which is preceeded by _run_ParameterSearch_.
For example, if you want to run the experiments for SpectralCF, you should run this command:
```Python
python run_ParameterSearch_RecSys_18_SpectralCF.py
```

The script will:
* Load and split the data.
* Run the bayesian hyperparameter optimization on all baselines, saving the best values found.
* Run the fit and test of the DL algorithm
* Create the latex code of the result tables, as well as plot the data splits, when required. 
* The results can be accessed in the _result_experiments_ folder.






## Installation

Note that this repository requires Python 3.6

First we suggest you create an environment for this project using virtualenv (or another tool like conda)

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment:

If you are using virtualenv:
```Python
virtualenv -p python3 DLevaluation
source DLevaluation/bin/activate
```
If you are using conda:
```Python
conda create -n DLevaluation python=3.6 anaconda
source activate DLevaluation
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

At this point you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```Python
python run_compile_all_cython.py
```

### Matlab engine
In addition to the repository dependencies, KDD CollaborativeDL also requires the Matlab engine, due to the fact that the algorithm is developed in Matlab. 
To install the engine you can use a script provided directly with your Matlab distribution, as described in the [Matlab Documentation](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
The algorithm requires also a GSL distribution, whose installation folder can be provided as a parameter in the fit function of our Python wrapper. Please refer to the original [CollaborativeDL README](Conferences/KDD/CollaborativeDL_github_matlab/README.md) for all installation details.

 