from pickletools import optimize
from pyexpat import model
from statistics import mode
from turtle import shape, update
from unicodedata import name
import keras 
import os
import tensorflow as tf
import numpy as np
from keras.layers import Input,Dense,Reshape,Flatten,Activation,MaxPool2D
from keras.layers import BatchNormalization,Conv2DTranspose,Conv2D,Dropout
from keras.models import Sequential,Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
import cv2
from keras.datasets import mnist

img_rows = 28
img_cols = 28
img_channel = 3
img_shape = (img_rows,img_cols,img_channel)
img_dir = '/home/azadm/Desktop/Datasetf_For_ML/train/train/'
file_name = "test_img/test.png"


optimizer = Adam(0.001, 0.5) # lr and momentum

def build_generator():

    noise_shape = (100,) #generator takes 1D array of size 100 as Input

    model = Sequential(name="Generator")
    model.add(Dense(256,input_shape=noise_shape, name="Input_layer"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, name="Dense_1"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024, name="Dense_2"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod(img_shape),activation="tanh", name="Dense_3"))
    model.add(Reshape(img_shape, name="Reshape_layer"))

    # model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise,img)

generator = build_generator()
generator.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

def build_discriminator():
    
    model = Sequential(name="Discriminator")

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1,activation="sigmoid"))
    
    # model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img,validity)

discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

def GAN(generator,discriminator):
    discriminator.trainable = False
    # Connet generator and discriminator
    model = Sequential(name="GAN")
    model.add(generator)
    model.add(discriminator)
    return model

gan = GAN(generator,discriminator)
gan.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])
# print(gan.summary())

def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (img_rows, img_cols),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

def train(epochs, batch_size=128, save_interval=500):

    x_train, class_name = create_dataset(img_dir)

    # (x_train, _) , (_, _) = mnist.load_data()
    # x_train = (x_train.astype(np.float32)-127.5)/127.5
    #add channels dimention. As the input to our gen and discriminator. has a shape of 28x28x1
    # x_train = np.expand_dims(x_train,axis=3)

    x_train = np.array(x_train)
    half_batch = int(batch_size/2)
    generator.load_weights('generator_model.h5')
    discriminator.load_weights('discriminator_model.h5')
    gan.load_weights('gan_model.h5')

    for epoch in range(epochs):
        # ------------------------------
        # Train the discrimiunator model
        # ------------------------------
        # Select a random half batch of fake images

        idx = np.random.randint(0,x_train.shape[0],half_batch)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, (half_batch,100))
        # GEnerate a half batch of fake images
        gen_images = generator.predict(noise)

        #Train the discriminator on real and fake images, seperately
        d_loss_real = discriminator.train_on_batch(imgs,np.ones((half_batch,1)))
        d_loss_fake = discriminator.train_on_batch(gen_images,np.zeros((half_batch,1)))

        d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

        noise = np.random.normal(0,1,(batch_size,100))
        #creates an array of all ones of size=batch size
        valid_y = np.array([1] * batch_size)

        g_loss = gan.train_on_batch(noise,valid_y)

        print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

        if epoch % save_interval == 0:
            save_imgs(epoch)

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0,1,(r * c, 100))
    gen_imgs = generator.predict(noise)

    # Rescale imgae 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt+=1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()

train(epochs=50000, batch_size=32, save_interval=1000)

generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')
gan.save('gan_model.h5')

#Generator model
generator.load_weights('generator_model.h5')
layers = (generator.layers)[1].layers

gen_model = Sequential(name = 'Updated_generator_model')

for layer in (generator.layers)[1].layers[:-2]:
    gen_model.add(layer)

gen_model.add(Dense(np.prod(img_shape)))
gen_model.add(Dense(14*14*3))
gen_model.add(Reshape((14,14,3)))
gen_model.add(Conv2DTranspose(3,(3,3), strides=(2,2), padding='same'))
gen_model.add(Activation("sigmoid"))

gen_model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

#Discriminator model
discriminator.load_weights('discriminator_model.h5')
layers = (discriminator.layers)[1].layers

dcr_model = Sequential(name = 'Updated_dcr_model')

dcr_model.add(Conv2D(128,(3,3),strides=(2,2),padding='same',input_shape=img_shape))
dcr_model.add(LeakyReLU(alpha=0.2))
dcr_model.add(Conv2D(64,(3,3),strides=(2,2),padding='same'))
dcr_model.add(LeakyReLU(alpha=0.2))
dcr_model.add(Conv2D(32,(3,3),strides=(2,2),padding='same'))
dcr_model.add(LeakyReLU(alpha=0.2))

dcr_model.add(Flatten())
dcr_model.add(Dropout(0.4))
dcr_model.add(Dense(np.prod(img_shape)))
for layer in (discriminator.layers)[1].layers[1:]:
    dcr_model.add(layer)

dcr_model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

#Updated Gan model
updated_gan = GAN(gen_model,dcr_model)
updated_gan.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

updated_gan.summary()


def train_updated_model(epochs, batch_size=128, save_interval=500):

    x_train, class_name = create_dataset(img_dir)

    # (x_train, _) , (_, _) = mnist.load_data()
    # x_train = (x_train.astype(np.float32)-127.5)/127.5
    #add channels dimention. As the input to our gen and discriminator. has a shape of 28x28x1
    # x_train = np.expand_dims(x_train,axis=3)

    x_train = np.array(x_train)
    half_batch = int(batch_size/2)
    gen_model.load_weights('updated_generator_model.h5')
    dcr_model.load_weights('updated_discriminator_model.h5')
    updated_gan.load_weights('updated_gan_model.h5')

    for epoch in range(epochs):
        # ------------------------------
        # Train the discrimiunator model
        # ------------------------------
        # Select a random half batch of fake images

        idx = np.random.randint(0,x_train.shape[0],half_batch)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, (half_batch,100))
        # GEnerate a half batch of fake images
        gen_images = gen_model.predict(noise)

        #Train the discriminator on real and fake images, seperately
        d_loss_real = dcr_model.train_on_batch(imgs,np.ones((half_batch,1)))
        d_loss_fake = dcr_model.train_on_batch(gen_images,np.zeros((half_batch,1)))

        d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

        noise = np.random.normal(0,1,(batch_size,100))
        #creates an array of all ones of size=batch size
        valid_y = np.array([1] * batch_size)

        g_loss = updated_gan.train_on_batch(noise,valid_y)

        print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

        if epoch % save_interval == 0:
            save_imgs(epoch)

# train_updated_model(10000,32,100)

# gen_model.save('updated_generator_model.h5')
# dcr_model.save('updated_discriminator_model.h5')
# updated_gan.save('updated_gan_model.h5')






    