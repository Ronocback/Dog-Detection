import os
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split

image_path = 'images/Images/'
breeds = os.listdir(image_path)

breeds = breeds[:25]

random_seed = 66

num_labels = len(breeds)

def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        for image_name in os.listdir(image_path+"/"+category):
            img = cv2.imread(image_path+"/"+category+"/"+image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_array = Image.fromarray(img, 'RGB')
            
            #resize image to 227 x 227 because the input image resolution for AlexNet is 227 x 227
            resized_img = img_array.resize((227, 227))
            
            img_lst.append(np.array(resized_img))
            
            labels.append(index)
    return img_lst, labels

images, labels = load_images_and_labels(breeds)
print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))

images = np.array(images)
labels = np.array(labels)

n = np.arange(images.shape[0])

np.random.seed(random_seed)
np.random.shuffle(n)

print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = random_seed)

