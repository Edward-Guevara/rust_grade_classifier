from spectral import envi, save_rgb, imshow
import numpy as np
import matplotlib.pyplot as plt
import glob
import functions as fun

"""
Introduce the folder name, sample name of hsi to calibrate and a root folder
"""

folder = "testing"
sample = "15L_20R_day"

root_folder =  "C:/Users/Edward/OneDrive/OneDrive - Universidad Tecnológica de Panamá/Projects"

hdr = glob.glob(root_folder + '/rust_grade_classifier/data_ingestion/samples/{}/{}/*.hdr'.format(folder, sample))                                                     
Bin = glob.glob(root_folder + '/rust_grade_classifier/data_ingestion/samples/{}/{}/*.bin'.format(folder, sample))  

open_data = envi.open(hdr[0], Bin[0])
data = np.array(open_data.load())

bands =  np.arange(400, 1105, 5)

"""
HSI Visualization through RGB composition 
"""
plt.figure()
imshow(data, (49, 71, 89))
plt.title('HSI and white reference patch')


"""
calibrations of hyperespectral images
"""

if folder == "training":
    hsi_raw = data[150:, 150:550, :]
    white_ref = {"5_day": data[54:104,330:380, :],
                 "10_day": data[56:107, 296:347, :],
                 "15_day": data[56:106, 296:346, :],
                 "20_day": data[63:113, 283:333, :]}
else:
    hsi_raw = data[110:401, :601, :]
    white_ref = {"5L_10R_day": data[30:81, 275:326, :],
                 "15L_20R_day":  data[30:81, 255:306, :]}

"""
sample RGB visualizations
"""

plt.figure()
imshow(hsi_raw, (49, 71, 89))
plt.title('HSI Raw\n RGB')

"""
white reference RGB visualizations
"""

plt.figure()
imshow(white_ref[sample], (49, 71, 89))
plt.title('white reference')

mean_spectrum = np.mean(np.mean(hsi_raw, axis = 1), axis = 0)
mean_ref = np.mean(np.mean(white_ref[sample], axis = 1), axis = 0)


"""
white reference curve 
"""
plt.figure()
plt.plot(bands, mean_ref)
plt.title('White reference')

"""
mean spectrum curve 
"""
plt.figure()
plt.plot(bands, mean_spectrum)
plt.title('HSI Mean Spectrum')
    

"""
hyperespectral data calibtation
"""
hsi_2D = hsi_raw.reshape(hsi_raw.shape[0]*hsi_raw.shape[1], hsi_raw.shape[2])
reflectance = np.divide(hsi_2D, mean_ref)
print('Calibrated HSI\n', 'range:', np.min(reflectance),'-', np.max(reflectance))

reflectance = np.where(reflectance > 1, 1, reflectance)

hsi = reflectance.reshape(hsi_raw.shape[0], hsi_raw.shape[1], hsi_raw.shape[2])

"""
mean spectrum curve calibrated
"""

plt.figure()
plt.plot(bands, np.mean(np.mean(hsi, axis = 1), axis = 0))
plt.title('calibrated Mean Spectrum')
    

print('Re-calibrated HSI\n', 'range:', np.min(hsi),'-', np.max(hsi))               
imshow(hsi, (49, 71, 89))

"""
Script to save calibrated hsi and image rbg
"""

folder_name = 'hsi/{}'.format(folder)

fun.make_folder(folder_name)

save_rgb(folder_name + "/hsi_{}".format(sample) + ".jpg", hsi, [49, 71, 89])
np.savez_compressed(folder_name + "/hsi_{}".format(sample), hsi)

