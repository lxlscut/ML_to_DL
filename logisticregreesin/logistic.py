import numpy as np
from sklearn.metrics import accuracy_score

class logis:
    def __init__(self):
        self.theta = None
        self.coef_ = None
        self.intercept_ = None

    def logsitc(self,t):
        return 1./(1.+np.exp(-t))

    def fit(self,x,y,eta = 0.1,n_iteraters = 1e4):
        x_b = np.column_stack([np.ones((len(x), 1)), x])
        initial_theta = np.zeros(x_b.shape[1])
        # 求解当前误差
        def J(theta,X,Y):
            y_hat = self.logsitc(X.dot(theta))
            try:
                return -np.sum(Y*np.log(y_hat)+(1-Y)*np.log(1-y_hat))/len(y)
            except:
                return float("inf")
        # 求解当前的梯度
        def DJ(theta,x,y):
            return x.T.dot(self.logsitc(x.dot(theta)) - y)  / len(x_b)
        # 开始梯度下降
        def gradient_descent(theta,x,y,eta = eta,n_iteraters = n_iteraters,epsilon = 1e-6):
            n_iterator = 0
            theta_new = theta
            while n_iterator<n_iteraters:
                last_theta = theta_new
                gradient = DJ(theta_new,x,y)
                theta_new = theta_new - gradient*eta
                if (abs(J(theta_new,x,y) - J(last_theta,x,y))<epsilon):
                    break
                n_iterator += 1
            return theta_new

        self.theta = gradient_descent(initial_theta,x = x_b,y = y)
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self

    def pridictsocre(self,x):
        x_b = np.column_stack([np.ones((len(x), 1)), x])
        return self.logsitc(x_b.dot(self.theta))

    def predict(self,x):
        result = self.pridictsocre(x)
        return np.array(result>0.5,dtype="int")

    def score(self,x,y):
        y_predict = self.predict(x)
        return accuracy_score(y,y_predict)

    def __repr__(self):
        return "logis()"
