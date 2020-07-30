import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


image_path = 'images/Images/'
breeds = os.listdir(image_path)

breeds = random.sample(breeds, 2)

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

images = images[n]
labels = labels[n]

images = images.astype(np.float32)
labels = labels.astype(np.int32)
images = images/255

print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = random_seed)

model=Sequential()

#1 conv layer
model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding="valid",activation="relu",input_shape=(227,227,3)))

#1 max pool layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(BatchNormalization())

#2 conv layer
model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid",activation="relu"))

#2 max pool layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(BatchNormalization())

#3 conv layer
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

#4 conv layer
model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

#5 conv layer
model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid",activation="relu"))

#3 max pool layer
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(BatchNormalization())


model.add(Flatten())

#1 dense layer
model.add(Dense(4096,input_shape=(227,227,3),activation="relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#2 dense layer
model.add(Dense(4096,activation="relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#3 dense layer
model.add(Dense(1000,activation="relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

#output layer
model.add(Dense(num_labels,activation="softmax"))

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

cb = [
    ModelCheckpoint(filepath='best_model.h5', monitor='accuracy', save_best_only=True)
]

model.fit(x_train, y_train, epochs=100, callbacks=cb)

model = load_model('best_model.h5')

loss, accuracy = model.evaluate(x_test, y_test)

pred = model.predict(x_test)

plt.figure(1 , figsize = (19 , 10))
n = 0 

for i in range(9):
    n += 1 
    r = np.random.randint( 0, x_test.shape[0], 1)
    
    plt.subplot(3, 3, n)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    plt.imshow(x_test[r[0]])

    dec_act = round(y_test[r[0]], 3)
    dec_pred = round(y_test[r[0]]*pred[r[0]][y_test[r[0]]], 3)

    plt.title('Actual = {} {}, Predicted = {} {}'.format(dec_act , breeds[int(round(dec_act))].split('-')[1], dec_pred, breeds[int(round(dec_pred))].split('-')[1]))
    plt.xticks([]) , plt.yticks([])

plt.show()
