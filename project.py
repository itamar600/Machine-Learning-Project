#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error


class CNN:

    def __init__(self, x_train, y_train, x_test, y_test, img_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.img_size = img_size
        self.model = self.model()

    def model(self):
        model = Sequential()
        model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(self.img_size, self.img_size, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.summary()
        return model

    def run(self):

        opt = Adam(lr=0.0001)
        self.model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        epochs = 100
        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, validation_data=(self.x_test, self.y_test))

        """#show results of CNN"""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        predictions = self.model.predict_classes(x_test)
        predictions = predictions.reshape(1, -1)[0]
        print(classification_report(self.y_test, predictions, target_names=['pedestrain (Class 0)', 'regular (Class 1)']))
        return predictions

class KNN:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        knn_1 = KNeighborsClassifier(n_neighbors=1)
        knn_1.fit(self.x_train, self.y_train)
        knn_2 = KNeighborsClassifier(n_neighbors=2)
        knn_2.fit(self.x_train, self.y_train)
        knn_3 = KNeighborsClassifier(n_neighbors=3)
        knn_3.fit(self.x_train, self.y_train)
        knn_4 = KNeighborsClassifier(n_neighbors=4)
        knn_4.fit(self.x_train, self.y_train)
        knn_5 = KNeighborsClassifier(n_neighbors=5)
        knn_5.fit(self.x_train, self.y_train)

        """#show results of KNN"""

        score1 = knn_1.score(self.x_test, self.y_test)
        score2 = knn_2.score(self.x_test, self.y_test)
        score3 = knn_3.score(self.x_test, self.y_test)
        score4 = knn_4.score(self.x_test, self.y_test)
        score5 = knn_5.score(self.x_test, self.y_test)
        print("1 neighbour: ", score1)
        print("2 neighbours: ", score2)
        print("3 neighbours: ", score3)
        print("4 neighbours: ", score4)
        print("5 neighbours: ", score5)

        plt.scatter(self.x_test[:, 1], knn_1.predict_proba(self.x_test)[:, 1])
        return knn_1

class logisticRegression:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        logisticRegr = LogisticRegression()
        result = logisticRegr.fit(self.x_train, self.y_train)

        """#show results of Logistic Regression"""

        score = logisticRegr.score(self.x_test, self.y_test)
        print(score)

        plt.scatter(self.x_test[:, 1], logisticRegr.predict_proba(self.x_test)[:, 1])
        return logisticRegr

class SVM:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        SVM = svm.SVC(kernel='linear', probability=True)  # Linear Kernel
        SVM.fit(self.x_train, self.y_train)

        """#show results of SVM"""

        y_pred = SVM.predict(self.x_test)
        print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))
        print("Precision:", metrics.precision_score(self.y_test, y_pred))
        print("Recall:", metrics.recall_score(self.y_test, y_pred))

        plt.scatter(self.x_test[:, 1], SVM.predict_proba(self.x_test)[:, 1])
        return SVM

class adaboost:

    def __init__(self, x_test, y_test, cnn, knn, lgr, svm):
        self.x_test = x_test
        self.y_test = y_test
        self.cnn = cnn
        self.knn = knn
        self.lgr = lgr
        self.svm = svm

    def run(self):
        cnn_preds = self.cnn
        cnn_acc = metrics.accuracy_score(self.y_test, cnn_preds)
        svm_preds = self.svm.predict(self.x_test)
        svm_acc = metrics.accuracy_score(self.y_test, svm_preds)
        knn_preds = self.knn.predict(self.x_test)
        knn_acc = metrics.accuracy_score(self.y_test, knn_preds)
        logistic_preds = self.lgr.predict(self.x_test)
        logistic_acc = metrics.accuracy_score(self.y_test, logistic_preds)

        """#Combine all models to see if it is improving results"""

        # predictions
        adaboost_preds = []

        for x in range(len(self.cnn)):
            preds = 0.
            if svm_preds[x] == 0:
                preds += svm_acc * -1
            else:
                preds += svm_acc * 1
            if knn_preds[x] == 0:
                preds += knn_acc * -1
            else:
                preds += knn_acc * 1
            if logistic_preds[x] == 0:
                preds += logistic_acc * -1
            else:
                preds += logistic_acc * 1
            if cnn_preds[x] == 0:
                preds += cnn_acc * -1
            else:
                preds += cnn_acc *1
            if preds >= 0:
                adaboost_preds.append(1)
            else:
                adaboost_preds.append(0)
            # preds = []
            # preds.append(svm_preds[x])
            # preds.append(knn_preds[x])
            # preds.append(logistic_preds[x])
            # preds.append(cnn_preds[x])
            # print("count")
            # print(preds.count)
            # adaboost_preds.append(max(set(preds), key=preds.count))

        """#results of the combined models"""

        print("Accuracy:", metrics.accuracy_score(self.y_test, adaboost_preds))
        print("Precision:", metrics.precision_score(self.y_test, adaboost_preds))
        print("Recall:", metrics.recall_score(self.y_test, adaboost_preds))

        target_names = ['pedestrian', 'regular']
        print(classification_report(self.y_test, adaboost_preds, target_names=target_names))

        print(mean_squared_error(self.y_test, adaboost_preds, multioutput='raw_values'))

        print(confusion_matrix(self.y_test, adaboost_preds))

        plt.scatter(x_test[:, 1], adaboost_preds)

def findErrors(x_test, y_test, algo,isCnn= False):
    lst = []
    if(isCnn):
        for idx, prediction, label in zip(enumerate(x_test), algo, y_test):
            if prediction != label:
                lst.append(idx[0])
    else:
        for idx, prediction, label in zip(enumerate(x_test), algo.predict(x_test), y_test):
            if prediction != label:
                lst.append(idx[0])
    # npArr = np.array(lst, dtype=int)
    print("list: ", lst)
    return np.array(lst, dtype=int)
    # np.save = ('svm.npy', npArr)

if __name__ == '__main__':
    # x_train = np.fromfile('x_train.dat', dtype=float)
    # y_train = np.fromfile('y_train.dat', dtype=float)
    # x_test = np.fromfile('x_test.dat', dtype=float)
    # y_test = np.fromfile('y_test.dat', dtype=float)
    x_train = np.load('files/x_train.npy')
    y_train = np.load('files/y_train.npy')
    x_test = np.load('files/x_test.npy')
    y_test = np.load('files/y_test.npy')
    img_size = 128

    print("\n CNN \n")
    CNN = CNN(x_train, y_train, x_test, y_test, img_size).run()


    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    # x_train.shape

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
    # x_test.shape

    print("\n KNN \n")
    KNN = KNN(x_train, y_train, x_test, y_test).run()

    print("\n Logistic Regression \n")
    LGR = logisticRegression(x_train, y_train, x_test, y_test).run()

    print("\n SVM \n")
    SVM = SVM(x_train, y_train, x_test, y_test).run()

    print("\n Adaboost \n")
    adb = adaboost(x_test, y_test, CNN, KNN, LGR, SVM)
    adb.run()

    np.save('cnn.npy', findErrors(x_test, y_test, CNN, True))
    np.save('knn.npy', findErrors(x_test, y_test, KNN))
    np.save('lgr.npy', findErrors(x_test, y_test, LGR))
    np.save('svm.npy', findErrors(x_test, y_test, SVM))
    # findErrors(x_test, y_test, KNN, 'knn.npy')
    # findErrors(x_test, y_test, LGR, 'lgr.npy')
    # findErrors(x_test, y_test, SVM, 'svm.npy')
