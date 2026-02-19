import pandas as pd
import numpy as np

def cyclic_encode(data : pd.DataFrame , column : str):
    max_val = np.max(data['colums'])
    data[columns + "_sin"] = np.sin(2*np.pi*data[column]/max_val)
    data[columns + "_cos"] = np.cos(2*np.pi*data[column]/max_val)
    return data
