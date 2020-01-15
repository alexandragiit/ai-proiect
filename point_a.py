import pandas as pd
import numpy as np

fileName = 'Dataset.csv'

def normalize(dataset):
    # normalizarea presupune ca toate valorile sa fie numerice si sa fie in [0, 1]
    upperLimit = 1
    return ((dataset-dataset.min())/(dataset.max()-dataset.min()))*upperLimit


def getDataSets(name):
    dataSet = pd.read_csv(name)
    # df['split'] = np.random.randn(df.shape[0], 1)
    # frac specifica fractiunea de randuri pe care sa le returnezi intr un sample random
    # deci frac=1 zice sa returnam toate randurile (random)
    dataSet = dataSet.sample(frac=1)

    # normalizam datele
    dataSet = normalize(dataSet)

    # impart datele in tempSet + validationSet, apoi tempSet in testSet + trainingSet
    msk = np.random.rand(len(dataSet)) <= 0.9
    tempSet = dataSet[msk]
    testSet = dataSet[~msk]
    msk = np.random.rand(len(tempSet)) <= 0.9
    trainingSet = tempSet[msk]
    validationSet = tempSet[~msk]
    return trainingSet, validationSet, testSet


# print(getDataSets(fileName)[0])
# print(getDataSets(fileName)[1])
# print(getDataSets(fileName)[2])
