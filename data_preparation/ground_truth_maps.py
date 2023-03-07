import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
import functions as fun

folder = 'training'

root_folder = "C:/Users/Edward/OneDrive - Universidad Tecnológica de Panamá/Projects"

cluster_file_paths = glob.glob('cluster/{}/*.npz'.format(folder))
mask_file_paths = glob.glob(root_folder + '/rust_grade_classifier/data_ingestion/masks/{}/*.npz'.format(folder))

maps = [np.load(file)['arr_0'] for file in cluster_file_paths]
masks = [np.load(file)['arr_0'] for file in mask_file_paths]

filter1 = [mask*maps for mask, maps in zip(masks, maps)]
filter2 = [np.zeros((mask.shape[0], mask.shape[1])) for mask in masks]
filter2 = [np.where(mask == 0, 6, mask) for mask in masks]
gt = [filter2 + filter1 for filter2, filter1 in zip(filter2, filter1)]

gt = [np.where(gt == 4, 3, gt) for gt in gt]
gt = [np.where(gt == 5, 3, gt) for gt in gt]
gt = [np.where(gt == 2, 1, gt) for gt in gt]
gt = [np.where(gt == 6, 4, gt) for gt in gt]

if folder == "training":
    gt[1] = np.where(gt[1] == 4, 3, gt[1])
    gt[3] = np.where(gt[3] == 1, 2, gt[3])
    
else:
    gt = [np.where(gt == 4, 3, gt) for gt in gt]

gt = [gt*mask for gt, mask in zip(gt, masks)]


for i in range(len(gt)):
    plt.imshow(gt[i], cmap = "Spectral")
    plt.colorbar()
    plt.show()
    print(np.unique(gt[i]))
    

file_name = ['sample_{}'.format(i+1) for i in range(len(gt))]
file_path = "gt/{}".format(folder)

fun.make_folder(file_path)

for i in range(len(file_name)):
    np.savez_compressed(file_path + '/' + file_name[i], gt[i])