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



# scaler = StandardScaler()
# scaled = scaler.fit(X).transform(X)
# scaled_x = scaler.fit(X).transform(x)
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# encoded_y = encoder.transform(y)
# dummy_Y = np_utils.to_categorical(encoded_Y)
# dummy_y = np_utils.to_categorical(encoded_y)

# model = gait.model()
#
# for i in range(config['epochs']):
#     dam = model.fit(scaled, dummy_y)
#     dam.history['acc']
#
#     a = model.predict_classes(scaled_x)
#     dummy_y
#     a - y.values
#     model.evaluate(scaled, y)
#     model.evaluate(a, y)
#
# #
# pipeline = pipeline.fit(X,Y)

# with open('./helloworld.pickle', 'wb') as handle:
#     pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('./helloworld.pickle', 'rb') as handle:
#     load=pickle.load(handle)

# gait.kfold(X,Y, pipeline)