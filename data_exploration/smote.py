import imblearn
import pandas as pd
import numpy as np
import functions as f
import glob
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns


root_folder = "C:/Users/Edward/OneDrive - Universidad Tecnológica de Panamá/Projects/rust_grade_classifier"
data_path = glob.glob(root_folder + "/data_preparation/dataset/*.csv")
df = pd.read_csv(data_path[0])


X = df.drop('target', axis=1)
y = df['target']
train_df, test_df = f.split_data(X, y, test_size = 0.3)

X_train = train_df.drop('target', axis=1)
y_train =  train_df['target']


under_proportion = {'0001_ninety_percentage': 0.9,
                    '0002_eighty_percentage': 0.8,
                    '0003_seventy_percentage': 0.7,
                    '0004_sixty_percentage': 0.6,
                    '0005_fivety_percentage': 0.5}

folder_name = list(under_proportion.keys())
proportion = list(under_proportion.values())


for (name, proportion) in zip(folder_name, proportion):
    
    f.make_folder(name)


    oversampling_strategy = {2: int(len(y_train[y_train == 2])*1.5), 
                             4: int(len(y_train[y_train == 4])*1.6)}


    undersampling_strategy = {1: int(len(y_train[y_train == 1])*proportion), 
                              3: int(len(y_train[y_train == 3])*(proportion-0.1))}


    train_df = f.smote_under_dataset(X_train, y_train, oversampling_strategy, undersampling_strategy)


    data_proportion = {'Traininig data': train_df['target'].value_counts(normalize=True)*100,
                       'Testing data': test_df['target'].value_counts(normalize=True)*100}


    proportion_df = pd.DataFrame(data_proportion) 
    proportion_df = proportion_df.rename(index = {3: 'Ligera', 1: 'Severa', 2: 'Moderada', 4: 'Saludable'})
    print(proportion_df)

    proportion_df.to_csv('new_data_proportion.csv')

    fig, ax = plt.subplots(figsize = (10.5, 8))
    sns.set_theme(style="darkgrid")
    sns.histplot(data = train_df, 
                  x = 'target',  stat = "frequency", 
                  shrink= 2.8)
    ax.set_ylabel("Clases")
    ax.set_xlabel("Proporcionalidad [%]")
    ax.grid(True)
    plt.show()
    
    Band = np.arange(400, 1105, 5)
    mean = train_df.groupby('target')[['B{}'.format(i+1) for i in range(141)]].mean()
    median = train_df.groupby('target')[['B{}'.format(i+1) for i in range(141)]].median()

    mean_df = pd.DataFrame(mean).transpose()
    median_df = pd.DataFrame(median).transpose()

    index = {'B{}'.format(i+1):Band[i] for i in range(141)}

    mean_df = mean_df.rename(index = index)
    median_df = median_df.rename(index = index)

    mean_df = mean_df.rename_axis("Band")
    median_df = median_df.rename_axis("Band")


    fig, ax = plt.subplots()
    sns.lineplot(data = mean_df, palette="tab10", linewidth=2.5)
    ax.set_title('Perfiles espectrales (promedio)')
    ax.set_ylabel('Reflectividad')
    ax.set_xlabel('Longuitud de onda [nm]')
    ax.grid(True)
    ax.legend()
    plt.show()


    fig, ax = plt.subplots()
    sns.lineplot(data = median_df, palette="tab10", linewidth=2.5)
    ax.set_title('Perfiles espectrales (mediana)')
    ax.set_ylabel('Reflectividad')
    ax.set_xlabel('Longuitud de onda [nm]')
    ax.grid(True)
    ax.legend()
    plt.show()

    
    train_df.to_csv(name + '/train_dataset.csv', index = False)
    test_df.to_csv(name + '/test_dataset.csv', index = False)