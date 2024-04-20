import streamlit as st
import numpy as np 
import pandas as pd
import os
import cv2
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,UpSampling2D,Concatenate,Input,Softmax
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint

EPOCHS=10
BATCH_SIZE=10
HEIGHT=256
WIDTH=256
N_CLASSES=13

# Preprocess image
def LoadImage(name, path):
    img = Image.open(os.path.join(path, name))
    img = np.array(img)
    
    image = img[:,:256]
    mask = img[:,256:]
    
    return image, mask

def bin_image(mask):
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    return new_mask

def getSegmentationArr(image, classes, width=WIDTH, height=HEIGHT):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, : , 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c ).astype(int)
    return seg_labels

def give_color_to_seg_img(seg, n_classes=N_CLASSES):
    
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

classes = 13

# DataGenerator
def DataGenerator(path, batch_size=BATCH_SIZE, classes=N_CLASSES):
    files = os.listdir(path)
    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i+batch_size]
            imgs=[]
            segs=[]
            for file in batch_files:
                image, mask = LoadImage(file, path)
                mask_binned = bin_image(mask)
                labels = getSegmentationArr(mask_binned, classes)

                imgs.append(image)
                segs.append(labels)

            yield np.array(imgs), np.array(segs)

# U-Net model
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = Input((HEIGHT,WIDTH,3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = Conv2D(13, (1, 1), padding="same", activation="sigmoid")(u4)
    model = Model(inputs, outputs)
    return model

# Preprocess input from user
def resize_images_in_folder(input_folder, output_folder, size=(256, 256)):

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Open the image
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Resize the image
                img_resized = img.resize(size)
                # Save the resized image to the output folder
                img_resized.save(os.path.join(output_folder, filename))

def add_black_image_to_right_of_images(folder_path, black_image_size=(256, 256)):
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Load the original image
            original_image = Image.open(os.path.join(folder_path, filename))

            # Create a black image of specified size
            black_image = Image.new("RGB", black_image_size, color=(0, 0, 0))

            # Get the size of the original image
            original_width, original_height = original_image.size

            # Create a new image with width equal to the sum of original width and black image width
            combined_width = original_width + black_image_size[0]
            combined_height = original_height
            combined_image = Image.new("RGB", (combined_width, combined_height))

            # Paste the original image onto the new image
            combined_image.paste(original_image, (0, 0))

            # Paste the black image to the right of the original image
            combined_image.paste(black_image, (original_width, 0))

            # Save the combined image, overwriting the old image
            combined_image.save(os.path.join(folder_path, filename))

# Load model
model = UNet()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
model.load_weights("/app/seg_model.hdf5")

# Def path
input_folder = "/app/test"
output_folder = "/app/test_resize"



st.write("# Cityscapes Auto Labelling Program")
image_file = st.file_uploader("Upload your image", type=['png','jpeg','jpg'])

if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    st.write(file_details)
    img = Image.open(image_file)
    with open(os.path.join("/app/test",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    st.success("File has been uploaded")

resize_images_in_folder(input_folder, output_folder)
add_black_image_to_right_of_images(output_folder)

# Predict the image
testrun_val_gen = DataGenerator(output_folder , batch_size=BATCH_SIZE)
imgs, segs = next(testrun_val_gen)
pred = model.predict(imgs)

# Show the predicted result
max_show = 1
for i in range(max_show):
    _p = give_color_to_seg_img(np.argmax(pred[i], axis=-1))
    st.image(_p)

# Delete all files and end session
files = glob.glob('/app/test/*')
for f in files:
    os.remove(f)
files = glob.glob('/app/test_resize/*')
for f in files:
    os.remove(f)

st.stop()
