import pandas as pd
import numpy as np
import glob
import functions as f
from sklearn.model_selection import train_test_split


folder = 'training'

hsi_file_path = glob.glob('hsi/{}/*.npz'.format(folder))
gt_file_path = glob.glob('gt/{}/*.npz'.format(folder))


target = [1, 2, 3, 4]
names = ['Severa', 'Moderada', 'Ligera', 'Saludable']

"""
HSI and grundtruth dataset
"""
hsi = [np.load(i)['arr_0'] for i in hsi_file_path]
gt = [np.load(i)['arr_0'] for i in gt_file_path]



"""dataset
"""
dataset = [f.dataset(names, target, gt, hsi) for gt, hsi in zip(gt, hsi)]
data = np.vstack(tuple([data[0] for data in dataset])) 
target = np.hstack(tuple([data[1] for data in dataset])) 
name = np.hstack(tuple([data[2] for data in dataset]))
"""
dataframe
"""

col = ["B{}".format(i+1) for i in range(hsi[0].shape[2])]
df = pd.DataFrame(data, columns = col)
df['target'] = target
df['name'] = name


to_save = 'dataset'
f.make_folder(to_save)
df.to_csv(to_save + "/rust_grade_dataset.csv", index= False)