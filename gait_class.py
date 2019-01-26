# ProjectName : classification
# FileName : gait_class
# Created on: 2019. 01. 26.AM 5:53
# Created by: KwanHoon Lee

import numpy as np
import pandas
import matplotlib.pyplot as plt
import itertools

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
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


    def record(self, result):
        nowtxt = datetime.now().strftime('%m-%d_%H_%M')
        fout_d = open(self.config['result_path'] + nowtxt + '.txt', 'a')
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

        self.record(result)

    def learn_predict(self, X, Y, x, y, pipeline):
        pipeline.fit(X, Y)

        yhat = list(pipeline.predict(x))
        y = list(y)

        return confusion_matrix(y, yhat)

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

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        nowtxt = datetime.now().strftime('%m-%d_%H_%M')

        plt.savefig(self.config['plt_path'] + title + "_"+ nowtxt + ".png")
        plt.clf()

