"""
Created on 26/10/2020

@author: Maurizio Ferrari Dacrema
"""
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Base.Evaluation.Evaluator import EvaluatorHoldout

from Base.NonPersonalizedRecommender import TopPop
from GraphBased.P3alphaRecommender import P3alphaRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender

"""
This script provides a simple example on how to use the implemented recommender models.
All algorithms have the same basic interface, the constructor takes as input only sparse matrices of scipy.sparse format.
All recommender take as first argument in the constructor the URM, content-based models also take the ICM or UCM as the second.
- User Rating Matrix (URM) of shape |n users|x|n items| containing the user-item interactions, either implicit (1-0) or explicit (any value)
- Item Content Matrix (ICM) of shape |n items|x|n item features| containing the item features, again with any numerical value
- User Content Matrix (UCM) of shape |n users|x|n users features| containing the item features, again with any numerical value
"""

# Use a dataReader to load the data into sparse matrices
data_reader = Movielens1MReader()
loaded_dataset = data_reader.load_data()

# In the following way you can access the entire URM and the dictionary with all ICMs
URM_all = loaded_dataset.get_URM_all()
ICM_dict = loaded_dataset.get_loaded_ICM_dict()

# Create a training-validation-test split, for example by leave-1-out
# This splitter requires the DataReader object and the number of elements to holdout
dataSplitter = DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=True)

# The load_data function will split the data and save it in the desired folder.
# Once the split is saved, further calls to the load_data will load the splitted data ensuring you always use the same split
dataSplitter.load_data(save_folder_path= "result_experiments/usage_example/data/")

# We can access the three URMs with this function and the ICMs (if present in the data Reader)
URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

ICM_dict = dataSplitter.get_loaded_ICM_dict()


# Now that we have the split, we can create the evaluators.
# The constructor of the evaluator allows you to specify the evaluation conditions (data, recommendation list length,
# excluding already seen items). Whenever you want to evaluate a model, use the evaluateRecommender function of the evaluator object
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5], exclude_seen=False)
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10, 20], exclude_seen=False)


# We now fit and evaluate a non personalized algorithm
recommender = TopPop(URM_train)
recommender.fit()

results_dict, results_run_string = evaluator_validation.evaluateRecommender(recommender)
print("Result of TopPop is:\n" + results_run_string)



# We now fit and evaluate a personalized algorithm passing some hyperparameters to the fit functions
recommender = P3alphaRecommender(URM_train)
recommender.fit(topK=100, alpha=0.5)

results_dict, results_run_string = evaluator_validation.evaluateRecommender(recommender)
print("Result of P3alpha is:\n" + results_run_string)


# We now use a content-based algorithm and a hybrid content-collaborative algorithm
ICM_genres = ICM_dict["ICM_genres"]
recommender = ItemKNNCBFRecommender(URM_train, ICM_genres)
recommender.fit(topK=100, similarity="cosine")

results_dict, results_run_string = evaluator_validation.evaluateRecommender(recommender)
print("Result of ItemKNNCBF is:\n" + results_run_string)



recommender = ItemKNN_CFCBF_Hybrid_Recommender(URM_train, ICM_genres)
recommender.fit(topK=100, similarity="cosine")

results_dict, results_run_string = evaluator_validation.evaluateRecommender(recommender)
print("Result of ItemKNN_CFCBF is:\n" + results_run_string)
