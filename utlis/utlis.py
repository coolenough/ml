import pandas as pd
import numpy as np

def cyclic_encode(data : pd.DataFrame , columns):
    max_val = np.max(data[columns])
    data[columns + "_sin"] = np.sin(2*np.pi*data[columns]/max_val)
    data[columns + "_cos"] = np.cos(2*np.pi*data[columns]/max_val)
    return data

def log_transform(data ; pd.DataFrame, columns):
    try:
        data[columns + "_log"] = np.log(data[columns])
    except Exception as e:
        raise Exception(f"The following Exception has occured {e}")
    return data;

def sqrt_transform(data : pd.DataFrame, columns):
    try:
        data[columns + "_sqrt"] = np.sqrt(data[columns])
    except Exception as e:
        raise Exception(f"The following Exception has occured {e}")
    return data

def Transformer(data : pd.DataFrame,func : str,columns):
    funcs = {
        "log" : log_transform,
        "sqrt" : sqrt_transform,
        "sin-cos" : cyclic_encode
        }
    if func not in funcs:
        raise Exception(f"{func} is not avilable please selct one from {funcs.keys()}")
    data = funcs[func](data , columns)
    return data
