# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from random import sample
import os
import numpy as np
from PIL import Image

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    print("\n Load:\n")
    x_train = np.load('files/x_train.npy')
    y_train = np.load('files/y_train.npy')
    x_test = np.load('files/x_test.npy')
    y_test = np.load('files/y_test.npy')
    x_train_for_image = np.load('files/x_train_for_image.npy')
    x_test_for_image = np.load('files/x_test_for_image.npy')
    cnn_errors = np.load('files/cnn.npy')
    knn_errors = np.load('files/knn.npy')
    lgr_errors = np.load('files/lgr.npy')
    svm_errors = np.load('files/svm.npy')
    print(x_train)
    for i in svm_errors:
        img = Image.fromarray(x_test_for_image[i])
        img.save('svm'+str(i)+'.png')
    #img.show()
    # files = os.listdir('data\pedestrian')
    # for file in sample(files, 863):
    #     os.remove(os.path.join('data\pedestrian',file))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
