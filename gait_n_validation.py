# ProjectName : 180125_footClassification
# FileName : gait_n_validation
# Created on : 30/01/201910:19 AM
# Created by : KwanHoon Lee

from gait_class import Gait
import numpy as np
file = "gait_n"

config = {
    "train": 80,
    "validation":0.2,
    "target": "Subject_Group",
    "targetNames": ["healthy", "Athlates", "Abnormal"],
    "shape": 9,
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

option_acc = {
    "title" : "Model Accuracy",
    "cost" : "acc",
}
option_loss = {
    "title" : "Model Loss",
    "cost" : "loss"
}

np.random.seed(config['seed'])

mean_confusion_matrix = np.zeros([3,3])

for i in range(10):
    gait = Gait(config)
    X, Y, x, y = gait.data()
    history, accuracy, confusion_matrix = gait.training_result(X, Y, x, y)

    mean_confusion_matrix += confusion_matrix

    gait.plot_history(history, option_acc, i)
    gait.plot_history(history, option_loss, i)

    gait.plot_confusion_matrix(confusion_matrix, title='Confusion matrix_' + str(i))
    gait.plot_confusion_matrix(confusion_matrix, normalize=True, title='Normalized confusion matrix_' + str(i))

mean_confusion_matrix = mean_confusion_matrix/5


gait.plot_confusion_matrix(mean_confusion_matrix , title='Mean Confusion Matrix')
gait.plot_confusion_matrix(mean_confusion_matrix , normalize=True, title='Mean Normalized Confusion Matrix')


