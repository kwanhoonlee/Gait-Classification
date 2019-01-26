# ProjectName : classification
# FileName : foot_classification
# Created on: 2018. 9. 17.PM 8:29
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
from datetime import datetime

# define wider model
def deep_model():
	# create model
	model = Sequential()
	model.add(Dense(num_columns*1, input_dim=num_columns, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_columns*1, kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(num_columns*1/2), kernel_initializer='normal', activation='relu'))
	# model.add(Dense(num_columns*4, kernel_initializer='normal', activation='sigmoid'))
	# model.add(Dense(num_columns*4, kernel_initializer='normal', activation='sigmoid'))
	# model.add(Dense(num_columns*2, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(3, activation="softmax"))
	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# plot_model(model, show_shapes=True, show_layer_names=True, to_file='deep_nn_' + nowtxt + '.png')
	return model


load_data = pandas.read_excel("Alldata_20180202.xlsx", header = 0 )
# X = load_data.iloc[:, 1:]
# Y = load_data['Subject_Group']

itr = load_data.shape[1] - 1
# for i in range(itr) :
for i in range(1):
    load_data = shuffle(load_data)
    X = load_data.iloc[:, 40:]
    Y = load_data['Subject_Group']

    nowtxt = datetime.now().strftime('%m-%d_%H_%M')
    fout_d = open('./result/sparse_categorical_crossentropy_BodySize&Gait/predictions_dm_' + nowtxt + '_' + str(i) + '.txt', 'a')

    num_columns = X.shape[1]

    seed = 7
    numpy.random.seed(seed)

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=deep_model, epochs=100, batch_size=3, verbose=2)))
    pipeline = Pipeline(estimators)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.3f%% (%.3f%%) " % (results.mean(), results.std()))
    fout_d.write(("Standardized: %.3f%% (%.3f%%)" % (results.mean()*100, results.std()*100)) + '\n')