# ===========K-Nearest Neighbors Model=========================================
#  *   K-Nearest Neighbors Algorithm.
# =============================================================================
#  *   Created By Eric Theodore Cornetto(Ida Bagus Dwi Putra Purnawa).
#  *   Github (https://github.com/EricCornetto).
# =============================================================================
#  *   GNU General Public License v3.0.
# =============================================================================
#             Python Machine Learning
# =============================================================================
import numpy as np
import math

class KNearestNeighbor():
    def fit(self,x_train,y_train,labels):
        self.x_train = x_train
        self.y_train = y_train
        self.labels = labels

    def distance(self,inputData):
        len_data = len(self.x_train)
        distanceX = np.array([])
        for i in range(len_data):
            x = self.x_train[i] - inputData[0]
            y = self.y_train[i] - inputData[1]
            xSQR = np.square(x)
            ySQR = np.square(y)
            result = xSQR + ySQR
            resultSQRT = math.sqrt(result)
            distanceX = np.append(distanceX,resultSQRT)
        return distanceX

    def predict(self,inputData):
        distanceX = self.distance(inputData)
        len_data = len(self.x_train)
        distanceXList = []
        for i in range(len_data):
            distanceXList.append(distanceX[i])
        kValues = []
        if len_data %2 == 0:
            for i in range(5):
                minValues = min(distanceXList)
                minIndex = distanceXList.index(minValues)
                kValues.append(self.labels[minIndex])
                distanceXList.remove(minValues)
        elif len_data %2 == 1:
            for i in range(2):
                minValues = min(distanceXList)
                minIndex = distanceXList.index(minValues)
                kValues.append(self.labels[minIndex])
                distanceXList.remove(minValues)
        return kValues
