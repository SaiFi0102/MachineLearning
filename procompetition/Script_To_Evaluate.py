# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:11:41 2018

@author: zaid
"""

import pandas as pd
import numpy as np

def evaluate_prediction_rmpse(Sales_Original,Sales_Predicted):
    intermediate=np.sum(np.square((Sales_Original-Sales_Predicted)/Sales_Original))
    rmpse=np.sqrt(intermediate/len(Sales_Original))
    return rmpse






