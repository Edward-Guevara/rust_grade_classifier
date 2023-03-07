import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as f
import seaborn as sns
import glob 


root_folder = "C:/Users/Edward/OneDrive - Universidad Tecnológica de Panamá/Projects/rust_grade_classifier"


data_path = glob.glob(root_folder + "/data_preparation/dataset/*.csv")
df = pd.read_csv(data_path[0])

"""
Exploration of training, validation and testing data
"""

X = df.iloc[:, :141].to_numpy()
y = df['name'].to_numpy()
train_df, test_df = f.split_data(X, y, test_size = 0.3)


"""
Stratification of dataset 
"""


fig, ax = plt.subplots(3, 1, figsize = (18, 12), sharex = True)
sns.set_theme(style = "darkgrid")

dataset = {'Conjunto de datos': df, 
           'Subconjuto de entrenamiento': train_df, 
           'Subconjuto de prueba': test_df}

for i in range(3):
    data = list(dataset.values())[i]
    data = data.rename(columns = {'target': 'Clases'})
    title = list(dataset.keys())[i]
    sns.histplot(data = data, x = 'B50' , hue = 'Clases', stat = 'frequency', ax = ax[i], palette = "tab10")
    ax[i].set_title(title)
    ax[i].set_xlabel('Rlefectividad')
    ax[i].grid(True)
plt.show()
    

"""
Unbalance of dataset
"""


fig, ax = plt.subplots(figsize = (10.5, 8))
sns.set_theme(style="darkgrid")
sns.histplot(data = train_df, 
             x = 'target',  stat = "percent", 
             shrink= 0.5)
ax.set_xlabel("Clases")
ax.set_ylabel("Proporcionalidad [%]")
ax.grid(True)
plt.show()

data_proportion = {'Training data': train_df['target'].value_counts(normalize=False),
                   'Testing data': test_df['target'].value_counts(normalize=False)}

proportion_df = pd.DataFrame(data_proportion, index = train_df['target'].unique())
print(proportion_df)

proportion_df.to_csv('data_proportion.csv')


"""
mean of training dataset
"""

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


"""
data distribution
"""

wl = np.arange(400, 1105, 100)
index = np.where(np.isin(Band, wl))[0]

fig, ax = plt.subplots(2, 4, figsize = (20.5, 11.5), sharey = True)
for n, (i, w) in enumerate(zip(index, wl)):
    if n <= 3:
        ax1 = ax[0, n]
        sns.boxplot(data = df, x = 'B{}'.format(i+1), y = 'name',  hue = 'name', palette="tab10", ax = ax1)
        ax1.set_ylabel('Clases')
        ax1.set_title('Banda {}: {} nm'.format(i+1, w))
        ax1.set_xlabel('Reflectividad')
        ax1.grid(True)
        ax1.legend()
    else:
        ax2 = ax[1, n-4]
        sns.boxplot(data = df, x = 'B{}'.format(i+1), y = 'name',  hue = 'name', palette="tab10", ax = ax2)
        ax2.set_ylabel('Clases')
        ax2.set_title('Banda {}: {} nm'.format(i+1, w))
        ax2.set_xlabel('Reflectividad')
        ax2.grid(True)
        ax2.legend()
plt.show()
         


