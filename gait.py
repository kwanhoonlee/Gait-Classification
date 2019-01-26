# ProjectName : classification
# FileName : gait_classification
# Created on: 2019. 01. 26.AM 5:53
# Created by: KwanHoon Lee

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from datetime import datetime

config = {"train":60 ,
          "test": 40,
          }

def deep_model():

    model = Sequential()
    model.add(Dense(num_columns, input_dim=num_columns, kernel_initializer='normal'))
    model.add(Dense(num_columns, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(int(num_columns*1/2), kernel_initializer='normal', activation= 'relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mae', 'acc'])

    return model

def record_result(i, result):
    nowtxt = datetime.now().strftime('%m-%d_%H_%M')
    fout_d = open('./result/k-fold/crossvalidation_score_' + nowtxt + '_' + str(i) + '.txt', 'a')
    fout_d.write(("Standardized: %.3f%% (%.3f%%)" % (result.mean()*100, result.std()*100)) + '\n')


# load_data = pandas.read_excel('./data/Gait_N_20190126.xlsx', header=0)
load_data = pandas.read_excel('./data/Gait_20190126.xlsx', header=0)

for i in range(1):

    load_data = shuffle(load_data)
    X = load_data.iloc[:config['train'], 1:]
    Y = load_data['Subject_Group'][:config['train']]

    x = load_data.iloc[config['train']:, 1:]
    y = load_data['Subject_Group'][config['train']:]
    num_columns = X.shape[1]
    seed = 7
    numpy.random.seed(seed)

    # TODO: Need to standardization
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=deep_model, epochs=100, batch_size=1, verbose=2)))
    pipeline = Pipeline(estimators)

    # to find hyperparameters
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    result = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.3f%% (%.3f%%) " % (result.mean(), result.std()))

    record_result(i, result)

    history = pipeline.fit(X, Y)

    pred = history.predict(x)
    preds = list(pred)
    y = list(y)
    confusion_matrix(y, preds)
