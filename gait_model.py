from gait_class import Gait

file = "gait"

config = {
          "train":60,
          "target":"Subject_Group",
          "targetNames":["healthy","Athlates","Abnormal"],
          "shape":27,
          "seed":7,
          "optimizer":"adam",
          "init":"glorot_uniform",
          "loss":"categorical_crossentropy",
          "epochs":250,
          "filename":"./data/" + file + "_20190126.xlsx",
          "result_path":"./result/" + file + "/k-fold/" + file + "_",
          "model_path":"./result/" + file + "/model/" + file + "_",
          "plt_path":"./result/" + file + "/plt/" + file + "_"
          }


for i in range(10):
    gait = Gait(config)
    X, Y, x, y = gait.data()
    pipeline = gait.pipeline()
    confusion_matrix, accuracy = gait.learn_predict(X, Y, x, y, pipeline, i)

    gait.plot_confusion_matrix(confusion_matrix, title='Confusion matrix_' + str(i)+ "_" + str(round(accuracy,3)))
    gait.plot_confusion_matrix(confusion_matrix, normalize=True, title='Normalized confusion matrix_'+ str(i)+ "_" + str(round(accuracy,3)))