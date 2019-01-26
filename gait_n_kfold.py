from gait_class import Gait

file = "gait_n"

config = {
          "train":60,
          "target":"Subject_Group",
          "targetNames":["healthy","Athlates","Abnormal"],
          "shape":9,
          "seed":7,
          "optimizer":"adam",
          "init":"glorot_uniform",
          "loss":"categorical_crossentropy",
          "epochs":150,
          "filename":"./data/" + file + "_20190126.xlsx",
          "kfold_path":"./result/" + file + "/k-fold/" + file + "_",
          "model_path":"./result/" + file + "/model/" + file + "_",
          "plt_path":"./result/" + file + "/plt/" + file + "_"
          }

for i in range(10):
    gait = Gait(config)
    X, Y, x, y = gait.data()
    pipeline = gait.pipeline()
    gait.kfold(X, Y, pipeline)

    # confusion_matrix = gait.learn_predict(X, Y, x, y, pipeline)
    #
    # gait.plot_confusion_matrix(confusion_matrix, title='Confusion matrix, without normalization_' + str(i))
    # gait.plot_confusion_matrix(confusion_matrix, normalize=True, title='Normalized confusion matrix_'+ str(i))