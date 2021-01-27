from sklearn import svm
import numpy as np


class Classifier(object):

    def __init__(self):
        self.clf = svm.SVC(kernel=self._custom_kernel)
        self.is_trained = False

    def _custom_kernel(self, X, Y):
        return np.dot(X, np.transpose(Y))

    def fit(self, x, y):
        self.clf.fit(x, y)
        self.is_trained = True

    def predict(self, x):
        return self.clf.predict(x)
