from DataContainer import *
from readData import *
from trainModel import *
from evalModel import *
import pickle

imagesPath = 'data'
modelName = 'modelSaved'
containerFile = 'dataContainer.pkl'
modelFile = 'trainedModel'
display_images = False # Indicates if we need to read data
read_data = True  # Indicates if we need to read data
train_model = True  # Indicates if we need to read data

# Indicates if we need to read data or we just 'read' it from binary file
if read_data:
    dataContainer = DataContainer()
    readData(imagesPath, dataContainer, containerFile, display_images)
else:
    with open(containerFile, 'rb') as input:
        dataContainer = pickle.load(input)

if train_model:
    trainModel(dataContainer,modelFile)

evalModel(dataContainer, modelFile)
