from spectral import envi, save_rgb, imshow
import numpy as np
import fun as f 
import matplotlib.pyplot as plt
import matplotlib
import glob 

users = 'Edward' 
folder = 'training'
folder_day = '10_day'

hdr = glob.glob('C:/Users/{}/OneDrive - Universidad Tecnológica de Panamá/Datos adjuntos/GITTS/PROJECTS/LOGISTICA/Code/Corrosion/samples/{}/{}\*.hdr'.format(users, folder,folder_day))                                                     
Bin = glob.glob('C:/Users/{}/OneDrive - Universidad Tecnológica de Panamá/Datos adjuntos/GITTS/PROJECTS/LOGISTICA/Code/Corrosion/samples/{}/{}\*.bin'.format(users, folder, folder_day))  

open_data = envi.open(hdr[0], Bin[0])
data =  np.array(open_data.load())

bands =  np.arange(400, 1105, 5)
f.rgb_imag(data, (49, 71, 89), 'HSI and white reference patch')

"""
training
"""
hsi_raw = data[150:, 150:550, :]
# white_ref = data[54:104,330:380, :] # 5 dias
white_ref = data[56:107, 296:347, :] # 10 dias
# white_ref = data[56:106, 296:346, :] # 15 dias
# white_ref = data[63:113, 283:333, :] # 20 dias

"""
validations
"""
# hsi_raw = data[110:401, :601, :]
# # white_ref =  data[30:81, 275:326, :] #5L_10R
# white_ref =  data[30:81, 255:306, :] #15L_20R

f.rgb_imag(hsi_raw, (49, 71, 89), 'HSI Raw\n RGB')
f.rgb_imag(white_ref, (49, 71, 89), 'white reference')


ref = f.mean_spectrum(white_ref)
    
# f.plotting(bands, ref, 'Wavelength', 'Reflectance','White reference')
# f.plotting(bands, f.mean_spectrum(hsi_raw), 'Wavelength', 'Reflectance','HSI Mean Spectrum')


hsi_data_2D = f.hsi_2D(hsi_raw)
reflectance = np.divide(hsi_data_2D, ref)
print('Calibrated HSI\n', 'range:',np.min(reflectance),'-', np.max(reflectance))

for i in range(reflectance.shape[0]):
    for j in range(reflectance.shape[1]):
        if reflectance[i, j] >= 1:
            reflectance[i, j] = 1
                
hsi = reflectance.reshape(hsi_raw.shape[0], hsi_raw.shape[1], hsi_raw.shape[2])

# f.plotting(bands, f.mean_spectrum(hsi) , 'Wavelength', 'Reflectance','Re-calibrated Mean Spectrum')
print('Re-calibrated HSI\n', 'range:',np.min(hsi),'-', np.max(hsi))               
imshow(hsi, (49, 71, 89))


# directory = 'C:/Users/{}/OneDrive - Universidad Tecnológica de Panamá/Datos adjuntos/GITTS/PROJECTS/LOGISTICA/Code/Corrosion/hsi/{}/hsi_{}'.format(users, folder,folder_day)
# save_rgb(directory + '.jpg', hsi, [49, 71, 89])
# np.savez_compressed(directory , hsi)

