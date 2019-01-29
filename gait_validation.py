from gait_class import Gait
import numpy as np
file = "gait"

config = {
    "train": 80,
    "validation":0.2,
    "target": "Subject_Group",
    "targetNames": ["healthy", "Athlates", "Abnormal"],
    "shape": 27,
    "seed": 7,
    "optimizer": "adam",
    "init": "glorot_uniform",
    "loss": "categorical_crossentropy",
    "epochs": 250,
    "filename": "./data/" + file + "_20190126.xlsx",
    "result_path": "./result/" + file + "/k-fold/" + file + "_",
    "model_path": "./result/" + file + "/model/" + file + "_",
    "plt_path": "./result/" + file + "/plt/" + file + "_",
    "train_path": "./result/"+ file + "/training/" + file + "_",
    "validation_path":"./result/" + file+ "/validation/" + file + "_",
}

option = {
    "title" : "Training&Validation",
    "cost" : "acc",
}

np.random.seed(config['seed'])

for i in range(10):
    gait = Gait(config)
    X, Y, x, y = gait.data()
    history, accuracy, confusion_matrix = gait.training_result(X, Y, x, y)

    gait.plot_history(history, option, i)
