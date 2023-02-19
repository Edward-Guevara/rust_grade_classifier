from spectral import kmeans
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os
import functions as fun


"""
introduce the folder name , training or testing, and numbers of cluster
"""

folder = 'testing'
ncluster = 6

hsi_path = glob.glob("hsi/{}/*.npz".format(folder))

path = "hsi/{}".format(folder)
extension = ".npz"
file_name = fun.get_files_with_extension(path, extension)
print("Files:", file_name)


hsi = [np.load(i)['arr_0'] for i in hsi_path]

maps = [kmeans(hsi, nclusters = ncluster, max_iterations = 20)[0] for hsi in hsi]

for i in range(len(maps)):
    imag = plt.imshow(maps[i], cmap = 'Spectral')
    plt.colorbar(imag, shrink = 0.7)
    plt.show()

folder_name = 'cluster/{}'.format(folder)
fun.make_folder(folder_name)
    
for maps, file_name in zip(maps, file_name):
    np.savez_compressed(folder_name + "/{}".format(file_name), maps)