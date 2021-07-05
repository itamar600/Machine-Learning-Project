#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

"""#generate data"""

labels = ['pedestrian', 'regular']
img_size = 128

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

data = get_data('data')

"""#split to train and test"""

train,test = train_test_split(data, test_size=0.33, random_state=42)
print(train.shape)
print(test.shape)

"""#show the balance in the train"""

l = []
for i in train:
    if(i[1] == 0):
        l.append("pedestrian")
    else:
        l.append("regular")
sns.set_style('darkgrid')
sns.countplot(l)

"""#show the balance in the test"""

r = []
for i in test:
    if(i[1] == 0):
        r.append("pedestrian")
    else:
        r.append("regular")
sns.set_style('darkgrid')
sns.countplot(r)

"""#dislpay some images from the data"""

plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])

"""#preper the data"""

x_train_list = []
y_train = []
x_test_list = []
y_test = []

for feature, label in train:
  x_train_list.append(feature)
  y_train.append(label)

for feature, label in test:
  x_test_list.append(feature)
  y_test.append(label)

# Normalize the data
x_train = np.array(x_train_list)/255
x_test = np.array(x_test_list)/255
# img = Image.fromarray(x_train[4])
# img.save('my.png')
# img.show()
x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""#do some random things on the data for better results"""

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
# img = Image.fromarray(x_train[4], 'RGB')
# img.save('my.png')
# img.show()
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
x_train_for_image = np.array(x_train_list)
x_test_for_image = np.array(x_test_list)
np.save('x_train_for_image.npy', x_train_for_image)
np.save('x_test_for_image.npy', x_test_for_image)
