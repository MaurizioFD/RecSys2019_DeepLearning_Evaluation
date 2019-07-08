#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/06/2018

@author: Maurizio Ferrari Dacrema
"""

def estimate_sparse_size(num_rows, topK):
    """
    :param num_rows: rows or colum of square matrix
    :param topK: number of elements for each row
    :return: size in Byte
    """

    num_cells = num_rows*topK

    # Size = 2*size(int32) + size(float64)
    sparse_size = 4*num_cells*2 + 8*num_cells

    return sparse_size