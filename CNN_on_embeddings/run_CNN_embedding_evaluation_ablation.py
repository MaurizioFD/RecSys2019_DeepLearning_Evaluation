"""
Created on 21/12/2019

@author: Maurizio Ferrari Dacrema
"""

from Base.DataIO import DataIO
import os, shutil
import numpy as np


def run_evaluation_ablation(recommender_class, recommender_input_args,
                            evaluator_test,
                            input_folder_path,
                            result_folder_path,
                            map_mode):




    recommender_object = recommender_class(*recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS)
    file_name_input = recommender_object.RECOMMENDER_NAME + "_best_model_last"
    file_name_output = recommender_object.RECOMMENDER_NAME

    if os.path.exists(result_folder_path + file_name_output + "_metadata.zip"):
        return


    result_folder_path_temp = result_folder_path + "__temp_model/"

    # If directory does not exist, create
    if not os.path.exists(result_folder_path_temp):
        os.makedirs(result_folder_path_temp)

    recommender_object.load_model(input_folder_path, file_name_input)
    recommender_object.save_model(result_folder_path_temp, file_name_output)

    # Alter saved object to force in the desired map mode
    dataIO = DataIO(folder_path = result_folder_path_temp)
    data_dict = dataIO.load_data(file_name = file_name_output)

    data_dict["map_mode"] = map_mode
    dataIO.save_data(file_name = file_name_output, data_dict_to_save = data_dict)


    recommender_object = recommender_class(*recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS)
    recommender_object.load_model(result_folder_path_temp,
                                  file_name=file_name_output)

    results_dict, results_run_string = evaluator_test.evaluateRecommender(recommender_object)

    shutil.rmtree(result_folder_path_temp, ignore_errors=True)

    result_file = open(result_folder_path + file_name_output + ".txt", "w")
    result_file.write(results_run_string)
    result_file.close()

    results_dict = {"result_on_last": results_dict}

    dataIO = DataIO(folder_path = result_folder_path)
    dataIO.save_data(file_name = file_name_output + "_metadata",
                     data_dict_to_save = results_dict)
