import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


l = "esp"

def preprocess(l, path):
    ids = []
    train_data = pd.read_csv(path)
    scores = train_data["Score"]
    train_data.drop(["Score"], axis = 1, inplace  = True)
    cell_value_train = train_data.iloc[0, train_data.columns.get_loc('Text')]
    print(repr(cell_value_train))
    split_text2 = train_data["Text"].str.split("\r\n", expand=True)  
    # print(split_text2)
    train_data["sent0"] = split_text2[0]
    train_data["sent1"] = split_text2[1]
    # train_data["Type"] = "train"
    # train_data["Lang"] = "esp"
    # train_data["Scores"] = scores
    
    ids = train_data['PairID'].to_list()
    train_data.drop(["PairID", "Text"], axis=1, inplace=True)
    train_data.to_csv(f'train_{l}.csv', index = False)
    return train_data, ids

train_path_esp = "{}/{}_train.csv".format(l, l)
train_data_esp, ids = preprocess(l, train_path_esp)
train_data_esp.head(10)