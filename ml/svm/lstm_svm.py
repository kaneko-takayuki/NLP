# -*- coding: utf-8 -*-

from ml.base import MLBases
from sklearn import svm


class LSTMSVM(MLBases):
    def __init__(self):
        MLBases.__init__(self)

        self.model = svm.SVC()

    def train(self):
        """
        1エポック分の学習を
        :return: 
        """