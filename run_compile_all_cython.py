#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/03/2019

@author: Maurizio Ferrari Dacrema
"""

import sys, glob, traceback, os
from CythonCompiler.run_compile_subprocess import run_compile_subprocess


if __name__ == '__main__':

    subfolder_to_compile_list = [
        "MatrixFactorization",
        "Base/Similarity",
        "SLIM_BPR",
    ]


    cython_file_list = []

    for subfolder_to_compile in subfolder_to_compile_list:
        cython_file_list.extend(glob.glob('{}/Cython/*.pyx'.format(subfolder_to_compile), recursive=True))


    print("run_compile_all_cython: Found {} Cython files in {} folders...".format(len(cython_file_list), len(subfolder_to_compile_list)))
    print("run_compile_all_cython: All files will be compiled using your current python environment: '{}'".format(sys.executable))


    save_folder_path = "./result_experiments/"
    log_file_path = save_folder_path + "run_compile_all_cython.txt"

    # If directory does not exist, create
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)


    log_file = open(log_file_path, "w")

    fail_count = 0

    for file_index, file_path in enumerate(cython_file_list):

        file_path = file_path.replace("\\", "/").split("/")

        file_name = file_path[-1]
        file_path = "/".join(file_path[:-1]) + "/"


        log_string = "Compiling [{}/{}]: {}... ".format(file_index+1, len(cython_file_list), file_name)
        print(log_string)

        try:
            run_compile_subprocess(file_path, [file_name])

            log_string += "PASS\n"
            print(log_string)
            log_file.write(log_string)
            log_file.flush()

        except Exception as exc:
            traceback.print_exc()

            fail_count += 1
            log_string += "FAIL: {}\n".format(str(exc))
            print(log_string)
            log_file.write(log_string)
            log_file.flush()


    log_string = "run_compile_all_cython: Compilation finished. "

    if fail_count != 0:
        log_string += "FAILS {}/{}.".format(fail_count, len(cython_file_list))
    else:
        log_string += "SUCCESS."

    log_string += "\nCompilation log can be found here: '{}'".format(log_file_path)

    print(log_string)
    log_file.write(log_string)
    log_file.close()