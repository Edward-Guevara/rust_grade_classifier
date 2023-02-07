from spectral import envi, save_rgb, imshow, kmeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob 
import os


folder = "testing"
ncluster = 7

hsi_path = glob.glob("hsi/{}/*.npz".format(folder))


def get_files_with_extension(path, extension):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(filename)
    return files

path = "hsi/{}".format(folder)
extension = ".npz"
file_name = get_files_with_extension(path, extension)
print("Files:", file_name)



hsi = [np.load(i)['arr_0'] for i in hsi_path]

maps = [kmeans(hsi, nclusters = ncluster, max_iterations = 20)[0] for hsi in hsi]

for i in range(len(maps)):
    imag = plt.imshow(maps[i], cmap = 'Spectral')
    plt.colorbar(imag, shrink = 0.7)
    plt.show()

folder_name = 'cluster/{}'.format(folder)

try:
    os.makedirs(folder_name)
except FileExistsError:
    print(f"The folder {folder_name} already exists.")
    
for maps, file_name in zip(maps, file_name):
    np.savez_compressed(folder_name + "/{}".format(file_name), maps)