import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from logistic import logis


def getdata():
    iris = datasets.load_iris()
    data = iris.get("data")
    target = iris.get("target")
    x_train = data[target < 2, :2]
    y_train = target[target < 2]
    return x_train, y_train
if __name__ == '__main__':
    x_train, y_train = getdata()
    logist = logis()
    a = logist.fit(x= x_train,y= y_train)
    print(a.intercept_,a.coef_)