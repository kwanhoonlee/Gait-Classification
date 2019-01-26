import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

config = {
      "train": 60,
      "seed":7 ,
      "optimizers" :['rmsprop', 'adam'],
      "inits" : ['glorot_uniform', 'normal', 'uniform'],
      "losses" : ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
      "epochs" : [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 ]
      }

def deep_model(optimizer='rmsprop', init='glorot_uniform', loss='categorical_crossentropy'):

    model = Sequential()
    model.add(Dense(num_columns, input_dim=num_columns, kernel_initializer=init))
    model.add(Dense(int(num_columns*(1/2)), kernel_initializer=init, activation='relu'))
    # model.add(Dense(int(num_columns*1/2), kernel_initializer='normal', activation= 'relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mse', 'acc'])

    return model

# load_data = pandas.read_excel('./data/Gait_N_20190126.xlsx', header=0)
load_data = pandas.read_excel('./data/Gait_20190126.xlsx', header=0)

load_data = shuffle(load_data)
X = load_data.iloc[:config['train'], 1:]
Y = load_data['Subject_Group'][:config['train']]

num_columns = X.shape[1]

model = KerasClassifier(build_fn=deep_model, verbose=0)

param_grid = dict(optimizer=config['optimizers'], epochs=config['epochs'], init=config['inits'], loss=config['losses'])
grid = GridSearchCV(estimator=model, param_grid=param_grid)

print("Start to learn")
grid_result = grid.fit(X,Y)

print("Best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with : %r" %(mean, stdev, param))