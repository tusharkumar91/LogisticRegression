"""
This script educates the usage of logistic regression on wine dataset 
"""
from numpy import loadtxt, where, std
from pylab import scatter, show, legend, xlabel, ylabel,plot,xlim, ylim
import random
import csv
import math
import numpy
import pandas as pd
import random
import time

"""
This class executes the logistic regression and provides the necessary infrastructure
to execute gradient ascent.
"""
class LogisticRegression:
    def __init__(self, X, Y, learningRate, toleranceLimit, maxIterations, L1Parameter, L2Parameter, verbose = False):
        self.features = numpy.ones((X.shape[0], (X.shape[1]+1)))
        self.features[:, 1:] = X                        
        self.weights = numpy.zeros((X.shape[1]+1, 1))
        self.tolerance = toleranceLimit
        self.Y = Y.reshape((Y.shape[0],1))
        self.maxIterations = maxIterations
        self.verbose = verbose
        self.learningRate = learningRate
        self.iteration = 0
        self.L1Parameter = L1Parameter
        self.L2Parameter = L2Parameter
        
    def getProbabilities(self):
        return (1/(1 + numpy.exp(-self.features.dot(self.weights))))
    
    def getLogLikelihood(self):
        probabilities = self.getProbabilities()
        logLikelihood = (self.Y * (numpy.log(probabilities+1e-20))) + (1-self.Y)*(numpy.log((1-probabilities+1e-20)))
        regularizedSum = self.L2Parameter * (numpy.nansum(self.weights * self.weights))
        regularizedSum = regularizedSum + self.L1Parameter * (numpy.nansum(numpy.fabs(self.weights)))
        return (-1 * numpy.nansum(logLikelihood)) + regularizedSum
        
    def getLogLikelihoodGradient(self):
        error = self.Y - self.getProbabilities()
        product = error * self.features
        gradientOfLikelihood = product.sum(axis = 0).reshape(self.weights.shape)
        regularizedL2Gradient = -2 * self.L2Parameter * (self.weights)
        regularizedL1Gradient = -self.L1Parameter * (numpy.sign(self.weights))
        return gradientOfLikelihood + regularizedL2Gradient + regularizedL1Gradient
    
    def runGradientAscent(self):
        previousCost = 0
        currentCost = self.getLogLikelihood()
        self.iteration = 1
        if self.verbose:
            print("Cost before GA = " + str(currentCost))
        while((numpy.fabs(currentCost - previousCost) > self.tolerance) and self.iteration < self.maxIterations):
            gradientOfLikelihood = self.getLogLikelihoodGradient()
            self.weights = self.weights + self.learningRate*gradientOfLikelihood
            previousCost = currentCost
            currentCost = self.getLogLikelihood()
            if self.verbose:
                print("Cost after GA Step " + str(self.iteration) + " = " + str(currentCost))
            self.iteration = self.iteration + 1
        
"""
Function to scale feature .
X = X - Mean(X) / Std(X)
"""
def scaleFeature(feature):
    mean = feature.mean(axis = 0)
    std = feature.std(axis=0)
    return (feature-mean)/std

"""
Unscaling the weights according to the mechanism we used 
to scale the features
"""
def unscaleWeights(weights, unscaledFeature):
    mean = unscaledFeature.mean(axis = 0)
    std = unscaledFeature.std(axis=0)
    params = weights.T[0]/numpy.hstack((1,std))
    params[0] = params[0] - (mean*weights.T[0][1:]/std).sum()
    return params

if __name__ == "__main__":
    dataSet = pd.read_csv("data/wine.csv")    
    Y = (dataSet.Type=='Red').values.astype(int)
    X = dataSet.loc[:,dataSet.columns[0:11]].values
    Xscaled = scaleFeature(X)
    startTime = time.time()
    #Have kept the regularized parameter there just so that it can be played with.
    #Numbers mentioned below are without usage of regularized parameters
    logisticRegression = LogisticRegression(Xscaled, Y, 0.01, 0.0001, 2000, 0, 0.1, verbose=True)
    logisticRegression.runGradientAscent()
    endTime = time.time()
    params = unscaleWeights(logisticRegression.weights, X)
    print(numpy.nansum(numpy.fabs(params)))
    print((numpy.nansum(params * params)))
    print("Iterations required = " + str(logisticRegression.iteration))
    print("Final Cost = " + str(logisticRegression.getLogLikelihood()))
    print("Time taken for Classifier = " + str(endTime - startTime))
    print(params)

    # -----------------------------Sample Output ----------------------------------------#
    #                  Iterations required for linear case = 423                         #
    #                  Final Cost for linear case= 214.437999023                         #
    #       [ -1.85590439e+03  -3.98321625e-01   6.20264880e+00  -2.61376230e+00         #
    #         -9.52177020e-01   2.22076853e+01   6.72047393e-02  -5.32640869e-02         #
    #         1.85531805e+03  -1.79851569e+00   3.06669592e+00   1.91840067e+00]         #
    #------------------------------------------------------------------------------------#

    