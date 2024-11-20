import numpy as np
import os
from PIL import Image
def complate_mean_std(path,channels):
    folder_path = path
    total_pixels = 0
    sum_normalized_pixel_values = np.zeros(channels)
    sum_squared_diff= np.zeros(channels)
    for root,dirs,files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                try:
                    image = Image.open(image_path)
                    image_array = np.array(image)
                    normalized_image_array = image_array/255.0
                    total_pixels += normalized_image_array.size
                    sum_normalized_pixel_values += np.sum(normalized_image_array,axis=(0,1))
                except Exception as e:
                    print(f'无法识别文件 {image_path}: {e}')
    mean = sum_normalized_pixel_values/total_pixels

    for root,dirs,files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                try:
                    image = Image.open(image_path)
                    image_array = np.array(image)
                    normalized_image_array = image_array/255.0
                    diff = (normalized_image_array-mean)**2
                    sum_squared_diff += np.sum(diff,axis=(0,1))
                except Exception as e:
                    print(f'无法识别文件 {image_path}: {e}')
    variance = sum_squared_diff/total_pixels

    return mean,variance
mean,variance = complate_mean_std('D:\PyCharmProject\LeNet\PetImages',3)
print(mean,variance)