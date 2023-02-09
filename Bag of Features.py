import cv2
import numpy as np
import os
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

OriginalAndFakePath = 'G:\Porashuna\Image Processing Training Data/Fake and Original/'
OriginalAndFakeImageNames = os.listdir('G:\Porashuna\Image Processing Training Data/Fake and Original')
OriginalImagePath = 'G:/Porashuna/Image Processing Training Data/Original/'
OriginalImageNames = os.listdir('G:/Porashuna/Image Processing Training Data/Original')
FakeImagePath = 'G:/Porashuna/Image Processing Training Data/Fake/'
FakeImageNames = os.listdir('G:/Porashuna/Image Processing Training Data/Fake')

Dico = []
# FakeDico = []
surf = cv2.SURF()

for i in range(len(OriginalAndFakeImageNames)):
    OriginalImg = cv2.imread((OriginalAndFakePath+OriginalAndFakeImageNames[i]))
    Kp , Des = surf.detectAndCompute(OriginalImg, None)

    for d in Des:
        Dico.append(d)

# for i in range(len(FakeImageNames)):
#     FakeImg = cv2.imread((FakeImagePath+FakeImageNames[i]))
#     FkKp , FkDes = surf.detectAndCompute(FakeImg, None)
#
#     for d in FkDes:
#         FakeDico.append(d)


k = 20
batch_size = 6
OriginalKmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(Dico)
# FakeKmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(FakeDico)


OriginalKmeans.verbose = False
# FakeKmeans.verbose = False

histo_list = []
targetClass = []

for i in range(len(OriginalImageNames)):
    OriginalImg = cv2.imread((OriginalImagePath + OriginalImageNames[i]))
    OrgKp, OrgDes = surf.detectAndCompute(OriginalImg, None)

    FakeImg = cv2.imread((FakeImagePath + FakeImageNames[i]))
    FkKp, FkDes = surf.detectAndCompute(FakeImg, None)

    OrgHisto = np.zeros(k)
    OrgNkp = np.size(OrgKp)

    FkHisto = np.zeros(k)
    FkNkp = np.size(FkKp)

    for d in OrgDes:
        idx = OriginalKmeans.predict([d])
        OrgHisto[idx] += 1.00/OrgNkp

    targetClass.append(1)
    histo_list.append(OrgHisto)

    for e in FkDes:
        idx = OriginalKmeans.predict([e])
        FkHisto[idx] += 1.00/FkNkp

    targetClass.append(0)
    histo_list.append(FkHisto)




def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

histo_list_array = np.array(histo_list)
histo_list_dataframe=pd.DataFrame(histo_list_array)
targetClassDataFrame = pd.DataFrame(targetClass)

clf = SVC(kernel='sigmoid')
logreg = LogisticRegression()
mlp = MLPClassifier(verbose=True,max_iter=600000)
dataSplit=np.array_split(histo_list_dataframe,3)
targetClassSplit=np.array_split(targetClassDataFrame,3)
accuracy_scores_logistic=list()
accuracy_scores_svm=list()
accuracy_scores_neural = list()
for i in range(len(dataSplit)):
    trainData=list(dataSplit)
    trainTargetData=list(targetClassSplit)
    testData = dataSplit[i]
    testTargetData = targetClassSplit[i]
    del trainData[-i]
    del trainTargetData[-i]
    trainData = pd.concat(trainData)
    trainTargetData = pd.concat(trainTargetData)
    logistic=logreg.fit(trainData.values,trainTargetData)
    svm = clf.fit(trainData.values,trainTargetData)
    neuralNet = mlp.fit(trainData.values, trainTargetData)
    predictionSvm = svm.predict(testData.values)
    predictionLogistic=logistic.predict(testData.values)
    predictionNeuralNet = neuralNet.predict(testData.values)
    accurate_list=np.array(testTargetData)
    accuracySvm = accuracy_metric(accurate_list,predictionSvm)
    accuracyLogistic = accuracy_metric(accurate_list,predictionLogistic)
    accuracyNeuralNet = accuracy_metric(accurate_list,predictionNeuralNet)
    accuracy_scores_logistic.append(accuracyLogistic)
    accuracy_scores_svm.append(accuracySvm)
    accuracy_scores_neural.append(accuracyNeuralNet)
print('Logistic Accuracy Scores:',accuracy_scores_logistic)
print('Logistic Accuracy Scores Mean:',np.mean(accuracy_scores_logistic))
print('SVM Accuracy Scores:',accuracy_scores_svm)
print('SVM Accuracy Scores Mean:',np.mean(accuracy_scores_svm))
print('Neural Network Accuracy Scores:',accuracy_scores_neural)
print('Neural Network Accuracy Scores Mean:',np.mean(accuracy_scores_neural))
