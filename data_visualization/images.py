import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import glob

root_folder = "C:/Users/Edward/OneDrive/OneDrive - Universidad Tecnológica de Panamá/Projects/rust_grade_classifier"

folder = 'training'
images_path = glob.glob(root_folder + '/data_preparation/hsi/{}/*.jpg'.format(folder))
gt_path = glob.glob(root_folder + '/data_preparation/gt/{}/*.npz'.format(folder))

images = [np.array(Image.open(imag)) for imag in images_path]
gt = [np.load(gt)['arr_0'] for gt in gt_path]
gt[1][150, 150] = 4
gt[2][150, 150] = 4

fig, axs = plt.subplots(2, 2, figsize = (10, 10))
for i in range(2):
    for j in range(2):
        if i == 0:
            ax = axs[i, j] 
            ax.imshow(images[j])
            ax.set_title('Muestra {}'.format(j+1))
        else:
            ax = axs[i, j] 
            ax.imshow(images[j+2])
            ax.set_title('Muestra {}'.format(j+3))
plt.show()


target = [0, 1, 2, 3, 4]
labels = ['Fondo', 'Severa', 'Moderada', 'Ligera','Saludable']


fig, axs = plt.subplots(2, 2, figsize = (14, 10))
color_maps = 'viridis'
for i in range(2):
    for j in range(2):
        if i == 0:
            ax = axs[i, j]
            colors = cm.get_cmap(color_maps, len(np.unique(gt[j])) + 1)
            p = ax.imshow(gt[j], cmap = colors, vmin = np.min(gt[j]) - 0.5, vmax = np.max(gt[j]) + 0.5)
            ax.set_title('Muestra {}'.format(j+1))
            fig.colorbar(p, ax = ax, shrink = 0.9).set_ticks(target[:len(np.unique(gt[j])) + 1], labels = labels[:len(np.unique(gt[j])) + 1])
        else:
            ax = axs[i, j] 
            colors = cm.get_cmap(color_maps, len(np.unique(gt[j+2])) + 1)
            p = ax.imshow(gt[j+2], cmap = colors, vmin = np.min(gt[j+2]) - 0.5, vmax = np.max(gt[j+2]) + 0.5)
            ax.set_title('Muestra {}'.format(j+3))
            fig.colorbar(p, ax = ax, shrink = 0.9).set_ticks(target[:len(np.unique(gt[j+2])) + 1], labels = labels[:len(np.unique(gt[j+2])) + 1])
plt.show()            