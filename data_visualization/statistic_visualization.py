import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import glob 


root_folder = "C:/Users/Edward/OneDrive - Universidad Tecnológica de Panamá/Projects/rust_grade_classifier"


data_path = glob.glob(root_folder + "/data_preparation/dataset/*.csv")
df = pd.read_csv(data_path[0])

"""
Exploration of training, validation and testing data
"""

def split_data(X, y, val_size = None, test_size = None, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size = val_size, 
                                                      random_state = random_state)
    
    col = ['B{}'.format(i+1) for i in range(141)]
    
    train_df = pd.DataFrame(X_train, columns = col)
    train_df['target'] = y_train
    
    val_df = pd.DataFrame(X_val, columns = col)
    val_df['target'] = y_val
    
    test_df = pd.DataFrame(X_test, columns = col)
    test_df['target'] = y_test
    
    
    return train_df, val_df, test_df


X = df.iloc[:, :141].to_numpy()
y = df['target'].to_numpy()
train_df, val_df, test_df = split_data(X, y, val_size = 0.17, test_size = 0.15)


"""
Stratification of dataset 
"""


fig, ax = plt.subplots(4, 1, figsize = (18, 12), sharex = True)
sns.set_theme(style = "darkgrid")

dataset = {'Conjunto de datos': df, 
           'Subconjuto de entrenamiento':train_df, 
           'Subconjuto de validacion':val_df, 
           'Subconjuto de prueba':test_df}

for i in range(4):
    
    sns.histplot(data = list(dataset.values())[i], x = 'B50' , hue = 'target', stat = 'frequency', ax = ax[i], palette="tab10")
    ax[i].set_title(list(dataset.keys())[i])
    ax[i].grid(True)
plt.show()
    

"""
Unbalance of dataset
"""


classes = {'Severa': 1, 'Moderada': 2, 'Ligera': 3, 'Saludable': 4}
names = list(classes.keys())
target = list(classes.values())

fig, ax = plt.subplots(figsize = (10.5, 8))
sns.set_theme(style="darkgrid")
sns.histplot(data = train_df, 
             y = 'target',  stat = "percent", 
             shrink= 3.5)
ax.set_ylabel("Clases")
ax.set_xlabel("Proporcionalidad [%]")
ax.grid(True)
ax.set_yticks(target, names)
plt.show()

data_proportion = {'Traininig data': np.unique(train_df['target'], return_counts = True)[1],
                   'Validation data': np.unique(val_df['target'], return_counts = True)[1],
                   'Testing data': np.unique(test_df['target'], return_counts = True)[1]}

proportion_df = pd.DataFrame(data_proportion, index = names) 
print(proportion_df)

proportion_df.to_csv('data_proportion.csv')


"""
mean of training dataset
"""

Band = np.arange(400, 1105, 5)

mean = train_df.groupby('target')[['B{}'.format(i+1) for i in range(141)]].mean()
mean_df = pd.DataFrame(mean).transpose()
mean_df = mean_df.rename(columns = {1: names[0], 2: names[1], 3: names[2], 4: names[3]})
index = {'B{}'.format(i+1):Band[i] for i in range(141)}
mean_df = mean_df.rename(index = index)
mean_df = mean_df.rename_axis("Band")

fig, ax = plt.subplots(figsize = (10.5, 8))
sns.lineplot(data = mean_df, palette="tab10", linewidth=2.5)
ax.set_ylabel('Reflectividad')
ax.set_xlabel('Longuitud de onda [nm]')
ax.grid(True)
ax.legend()
plt.show()




