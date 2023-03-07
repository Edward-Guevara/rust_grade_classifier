import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

def split_data(X, y, test_size = None, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)
    
    col = ['B{}'.format(i+1) for i in range(141)]
    
    train_df = pd.DataFrame(X_train, columns = col)
    train_df['target'] = y_train
    
    
    test_df = pd.DataFrame(X_test, columns = col)
    test_df['target'] = y_test
    
    return train_df, test_df


def smote_under_dataset(X_train, y_train, oversampling_strategy, undersampling_strategy):
    
    over = SMOTE(sampling_strategy = oversampling_strategy, random_state = 42)

    X_over, y_over = over.fit_resample(X_train, y_train)

    under = RandomUnderSampler(sampling_strategy = undersampling_strategy,  random_state = 42)

    X_under, y_under = under.fit_resample(X_over, y_over)

    train_df = pd.concat([X_under, y_under], axis = 1) 
    
    return train_df


def make_folder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        print(f"The folder {folder_name} already exists.")
