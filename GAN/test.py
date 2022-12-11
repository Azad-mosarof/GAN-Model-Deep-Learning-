from ctypes import sizeof
from multiprocessing.spawn import import_main_path
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os
import cv2

IMG_HEIGHT = 200
IMG_WIDTH = 200
dir = '/home/azadm/Desktop/Datasetf_For_ML/train/train/'


model = load_model('generator_model.h5')
file_name = "test_img/test.png"

# def create_dataset(img_folder):
   
#     img_data_array=[]
#     class_name=[]
   
#     for dir1 in os.listdir(img_folder):
#         for file in os.listdir(os.path.join(img_folder, dir1)):
       
#             image_path= os.path.join(img_folder, dir1,  file)
#             image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
#             image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
#             image=np.array(image)
#             image = image.astype('float32')
#             image /= 255 
#             img_data_array.append(image)
#             class_name.append(dir1)
#     return img_data_array, class_name

def save_imgs(file_name):
    r, c = 5, 5
    noise = np.random.normal(0,1,(r * c, 100))
    gen_imgs = model.predict(noise)

    # Rescale imgae 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5
    # img_data, class_name =create_dataset(dir)
    # gen_imgs = np.array(img_data[: r * c])

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt+=1
    fig.savefig(file_name)
    plt.close()

save_imgs(file_name=file_name)