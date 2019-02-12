# ProjectName : classification
# FileName : gait_class
# Created on: 2019. 01. 26.AM 5:53
# Created by: KwanHoon Lee

import numpy as np
import pandas
import matplotlib.pyplot as plt
import itertools
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from datetime import datetime

class Gait():

    def __init__(self, config):
        self.config = config


    def model(self):

        model = Sequential()
        model.add(Dense(self.config['shape'], input_dim=self.config['shape'], kernel_initializer=self.config['init']))
        model.add(Dense(int(self.config['shape']*(1/2)), kernel_initializer=self.config['init'], activation='relu'))
        # model.add(Dense(int(num_columns*1/2), kernel_initializer='normal', activation= 'relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss=self.config['loss'], optimizer=self.config['optimizer'], metrics=['mae', 'acc'])

        return model


    def record(self, result, path):

        nowtxt = datetime.now().strftime('%m-%d_%H_%M')
        fout_d = open(path + nowtxt + '.txt', 'a')
        fout_d.write(("Accuracy: %.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100)) + '\n')


    def data(self) :

        load_data = shuffle(pandas.read_excel(self.config['filename'], header=0))

        X = load_data.iloc[:self.config['train'], 1:]
        Y = load_data[self.config['target']][:self.config['train']]

        x = load_data.iloc[self.config['train']:, 1:]
        y = load_data[self.config['target']][self.config['train']:]

        return X, Y, x, y


    def pipeline(self):

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=self.model, epochs=self.config['epochs'], batch_size=1, verbose=2)))

        return Pipeline(estimators)


    def kfold(self, X, Y, pipeline):

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.config['seed'])
        result = cross_val_score(pipeline, X, Y, cv=kfold, scoring='accuracy')

        print("Accuracy: %.3f%% (%.3f%%) " % (result.mean() * 100, result.std() * 100))

        self.record(result, self.config['kfold_path'])


    def pipeline_fit_predict(self, X, Y, x, y, pipeline, i):

        pipeline_fit = pipeline.fit(X, Y)

        with open(self.config['model_path']+str(i)+"_"+datetime.now().strftime('%m-%d_%H_%M')+".pickle", 'wb') as handle:
            pickle.dump(pipeline_fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

        yhat = list(pipeline_fit.predict(x))
        y = list(y)

        accuracy = accuracy_score(y, yhat)

        print("Accuracy", accuracy)
        return confusion_matrix(y, yhat), accuracy


    def plot_confusion_matrix(self, cm,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        classes = self.config['targetNames']
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        nowtxt = datetime.now().strftime('%m-%d_%H_%M')

        plt.savefig(self.config['plt_path'] + title + "_"+ nowtxt + ".png", dpi=600)
        plt.clf()
        plt.close()


    def training_result(self, X, Y, x, y):

        scaler_X = StandardScaler()
        scaler_X.fit(X)
        X = scaler_X.transform(X)

        scaler_x = StandardScaler()
        scaler_x.fit(x)
        x = scaler_x.transform(x)

        model = self.model()
        Y, y = self.encoding(Y, y)

        history = model.fit(X, Y, validation_split=self.config['validation'], epochs=self.config['epochs'], batch_size=1, shuffle=True, verbose=0)

        yhat = model.predict_classes(x)
        accuracy = accuracy_score(y, yhat)
        print("Test Accuracy", accuracy)
        self.record(accuracy, self.config['train_path'])

        return history, accuracy, confusion_matrix(y, yhat)


    def encoding(self, Y, y):

        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        dummy_Y = np_utils.to_categorical(encoded_Y)

        encoded_y = encoder.transform(y)

        return dummy_Y, encoded_y


    def plot_training_history(self, history, title, i):

        nowtxt = datetime.now().strftime('%m-%d_%H_%M')

        plt.plot(history.history['acc'])
        plt.plot(history.history['loss'])
        plt.xlabel('Epoch')
        plt.legend(['Accuracy', 'Loss'], loc='lower left')
        plt.title(title)
        plt.savefig(self.config['plt_path'] + title + str(i)+ "_"+ nowtxt +".png", dpi=400)
        plt.clf()
        plt.close()

    def plot_history(self, history, option, i ):

        nowtxt = datetime.now().strftime('%m-%d_%H_%M')

        plt.plot(history.history[option['cost']])
        plt.plot(history.history["val_" + option['cost']])

        if option['cost'] == 'acc' :
            plt.ylabel('Accuracy')
            plt.legend(['train', 'validation'], loc='lower right')

        else :
            plt.ylabel('Loss')
            plt.legend(['train', 'validation'], loc='upper right')

        plt.xlabel('Epoch')
        plt.title(option['title'])

        plt.savefig(self.config['validation_path'] + option['title']+ "_" +str(i) + "_" + nowtxt +".png", dpi=600)

        plt.clf()
        plt.close()



