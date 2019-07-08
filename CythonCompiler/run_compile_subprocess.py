#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/2018

@author: Maurizio Ferrari Dacrema
"""

import subprocess, os, sys, shutil


def run_compile_subprocess(file_subfolder, file_to_compile_list):

    # Run compile script setting the working directory to ensure the compiled file are contained in the
    # appropriate subfolder and not the project root

    current_python_path = sys.executable

    compile_script_absolute_path = os.getcwd() + '/CythonCompiler/compile_script.py'
    file_subfolder_absolute_path = os.getcwd() + "/" + file_subfolder

    for file_to_compile in file_to_compile_list:

        try:
            command = [current_python_path,
                       compile_script_absolute_path,
                       file_to_compile,
                       'build_ext',
                       '--inplace'
                       ]

            output = subprocess.check_output(' '.join(command),
                                             shell=True,
                                             cwd=file_subfolder_absolute_path)


            try:

                command = ['cython',
                           file_to_compile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command),
                                                 shell=True,
                                                 cwd=file_subfolder_absolute_path)
            except:
                pass

        except Exception as exc:
            raise exc

        finally:
            # Removing temporary "build" subfolder
            shutil.rmtree(file_subfolder_absolute_path + "/build", ignore_errors=True)


    # Command to run compilation script
    # python CythonCompiler/compile_script.py filename.pyx build_ext --inplace

    # Command to generate html report
    # cython -a filename.pyx
