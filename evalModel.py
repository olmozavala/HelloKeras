from keras.models import load_model

def evalModel(dc, modelName):
    model = load_model(modelName)
    prediction = model.predict(dc.test)
    print(prediction)

