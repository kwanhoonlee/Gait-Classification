# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy
import pandas
from pandas import Series
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from keras.utils import plot_model
from keras.models import load_model
from sklearn.externals import joblib

from sklearn.utils import shuffle


# define wider model
def deep_model():
	# create model
	model = Sequential()
	model.add(Dense(num_columns*1, input_dim=num_columns, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_columns*1, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_columns*1, kernel_initializer='normal', activation='relu'))
	# model.add(Dense(num_columns*4, kernel_initializer='normal', activation='sigmoid'))
	# model.add(Dense(num_columns*4, kernel_initializer='normal', activation='sigmoid'))
	# model.add(Dense(num_columns*2, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	# plot_model(model, show_shapes=True, show_layer_names=True, to_file='deep_nn_' + nowtxt + '.png')
	return model

# load dataset

df_x = pandas.read_csv("x2_BP.csv", header=None, encoding="utf-8")
df_y = pandas.read_csv("y2.csv", header=None, encoding="utf-8")


print (df_x.head())
print (df_y.head())

itr = df_y.shape[1]

###
for i in range(itr):
	print (i)

	dataframe = pandas.concat([df_x, df_y[i]], axis=1)
	print (dataframe)

	print ('shuffling dataframe')
	dataframe = shuffle(dataframe)

	dataset = dataframe.values

	# split into train and test sets
	train_size = int(len(dataset) * 0.7)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	print(len(train), len(test))

	from datetime import datetime
	nowtxt = datetime.now().strftime('%m-%d_%H_%M')

	# fout = file('predictions_summary_' + nowtxt +'.txt' , 'w')
	# fout_s = file('predictions_s_' + now +'.txt' , 'w')
	# fout_l = file('predictions_l_' + now +'.txt' , 'w')
	# fout_w = file('predictions_w_' + now +'.txt' , 'w')
	fout_d = open('predictions_dm_' + nowtxt + '_' + str(i) +'.txt' , 'a')

	print (nowtxt)

	num_columns = len(dataset[0]) -1

	print ('Features:', num_columns)

	# split into input (X) and output (Y) variables
	X = dataset[:,0:num_columns]
	Y = dataset[:,num_columns]

	trainX = train[:,0:num_columns]
	trainY = train[:,num_columns]
	testX = test[:,0:num_columns]
	testY = test[:,num_columns]

	# # fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)

	print ('# evaluate model with deep model')
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=deep_model, epochs=1000, batch_size=2, verbose=0)))
	pipeline = Pipeline(estimators)

	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(pipeline, X, Y, cv=kfold)
	print("Deep: %.8f (%.8f) MSE" % (results.mean(), results.std()))
	fout_d.write(("Deep: %.8f (%.8f) MSE" % (results.mean(), results.std())) + '\n')

	## Save Model
	# trainX = StandardScaler().fit_transform(trainX)
	# dmodel = KerasRegressor(build_fn=deep_model, epochs=5000, batch_size=41, verbose=2)
	# dmodel.fit(trainX,trainY)
	#
	# dmodel.model.save('deep_model.h5')
	# dmodel.model.save_weights('deep_model_weights.h5')


	## fout
	# history = pipeline.fit(trainX, trainY)

	# preds = []
	# for i in range(len(testX)):
	# 	test = testX[i]
	# 	test = test.reshape(1,num_columns)
	# 	pred = history.predict(test)
	# 	preds.append(pred)
	# 	print pred, testY[i]

	# 	fout_d.write(str(pred) +'\t'+ str(testY[i]) + '\n')
	# r2 = r2_score(testY, preds)
	# print 'r2 :', r2
	# fout_d.write('\nR^2:\t'+ str(r2)+'\n')
