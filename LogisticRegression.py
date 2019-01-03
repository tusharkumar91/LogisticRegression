"""
This script educates the usage of logistic regression on dataset of marks 
where label informs if student was admitted or not.
We will first demonstrate the linear case where the features are marks X1 and X2 in both exams.
We will also demonstrate a linear case with higher order for the same data set,
by adding two features , X1^2 and X1*X2 in the data set and compare the results 
and decision boundary.
Plot is present in data folder to see the comparison
"""
from numpy import loadtxt, where, std
from pylab import scatter, show, legend, xlabel, ylabel,plot,xlim, ylim
import random
import csv
import math
import numpy
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
        self.iterations = 0
        self.L1Parameter = L1Parameter
        self.L2Parameter = L2Parameter
        
    def getProbabilities(self):
        return (1/(1 + numpy.exp(-self.features.dot(self.weights))))
    
    def getLogLikelihood(self):
        probabilities = self.getProbabilities()
        logLikelihood = (self.Y * (numpy.log(probabilities))) + (1-self.Y)*(numpy.log((1-probabilities)))
        regularizedSum = self.L2Parameter * (numpy.nansum(self.weights * self.weights))
        regularizedSum = regularizedSum + self.L1Parameter * (numpy.nansum(numpy.fabs(self.weights)))
        return (-1 * numpy.nansum(logLikelihood)) + regularizedSum
        
    def getLogLikelihoodGradient(self):
        error = self.Y - self.getProbabilities()
        product = error * self.features
        gradientOfLikelihood = product.sum(axis = 0).reshape(self.weights.shape)
        regularizedL2Gradient = 2 * self.L2Parameter * self.weights
        regularizedL1Gradient = self.L1Parameter * numpy.sign(self.weights)
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
Function to plot the decision boundary for the given dataSet.
Equations need to be modified for usage with other dataSets than the given dataSet
"""
def plotDecisionBoundary(dataSet, params, order = 1):
    randX = []
    numberOfFeatures = len(dataSet[0])
    #Generate 100 random numbers for X1 and then use the fact that w0 + Sum(wiXi) = 0 to get X2 
    for i in range(100):
        randX.append(random.random() * 100)
    
    #In case the data set is modified , 
    #these equations will change as there will be a change in number of features
    if order is not 1:
        # w0 + w1X1 + w2X2 + w3X1^2 + w4X1X4 = 0 at boundary
        randY = [ (-params[0] - (params[1] * x) - (params[3] * x * x))/(params[2] + (params[4]*x))  for x in randX]
    else:
        # w0 + w1X1 + w2X2 = 0 at boundary
        randY = [ (-params[0] - (params[1] * x))/(params[2])  for x in randX]
        
    Y = (dataSet[:,numberOfFeatures-1])
    X = dataSet[:,0:numberOfFeatures-1]
    pos = where(Y == 1)
    neg = where(Y == 0)
    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='y')
    if order is not 1:
        scatter(randX, randY, marker='+', c='r')
    else:
        scatter(randX, randY, marker='.', c='c')
    xlim(0,100)
    ylim(0,150)
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Not Admitted', 'Admitted', 'Linear Boundary'])

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


"""
Function to partition the dataSet into train and testData.
testRatio -> ratio of test examples:dataSet size. Default = 0.2
shouldShuffle -> whether the data needs to be shuffled or not. Default = false
"""
def partitionTrainAndTest(dataSet, testRatio = 0.2, shouldShuffle = True):
    dataSize = len(dataSet)
    if shouldShuffle:
        random.shuffle(dataSet)
    trainDataSize = int(dataSize*(1-testRatio))
    trainData = dataSet[:trainDataSize]
    testData = dataSet[trainDataSize:]
    return (trainData,testData)

"""
Function to get the accuracy for both linear and non-linear cases
"""
def getAccuracy(params, testData, order=1):
    numberOfFeatures = len(dataSet[0])
    Y = (dataSet[:,numberOfFeatures-1])
    X = dataSet[:,0:numberOfFeatures-1]
    errorCount = 0
    for index in range(len(testData)):
        if order is 1:
            likelihoodTerm = params[0]+(params[1]*(X[index][0])) + (params[2]*(X[index][1]))
        else:
            likelihoodTerm = params[0]+(params[1]*(X[index][0])) + (params[2]*(X[index][1])) + (params[3]*(X[index][0])*(X[index][0])) + (params[4]*(X[index][0])*(X[index][1]))
        if likelihoodTerm > 0 and Y[index] == 0:
            errorCount = errorCount + 1
        if likelihoodTerm <= 0 and Y[index] == 1:
            errorCount = errorCount + 1
    return (1 - (errorCount/float(len(testData))))*100


def showLinearExample(trainData, testData, dataSet):
    numberOfFeatures = len(dataSet[0])
    Y = (trainData[:,numberOfFeatures-1])
    X = trainData[:,0:numberOfFeatures-1]

    Xscaled = scaleFeature(X)
    startOfLinearCase = time.time()
    logisticRegression = LogisticRegression(Xscaled, Y, 0.01, 0.0001, 2000, 0.1, 0.1,verbose=False)
    logisticRegression.runGradientAscent()
    endOfLinearCase = time.time()
    params = unscaleWeights(logisticRegression.weights, X)
    print("Iterations required for linear case = " + str(logisticRegression.iteration))
    print("Final Cost for linear case= " + str(logisticRegression.getLogLikelihood()))
    print("Accuracy of linear Classifier = " + str(getAccuracy(params, testData)))
    print("Time taken for linear Classifier = " + str(endOfLinearCase - startOfLinearCase))
    plotDecisionBoundary(dataSet, params)

def showLinearExampleWithHigherOrder(trainData, testData, dataSet):
    numberOfFeatures = len(dataSet[0])
    Y = (trainData[:,numberOfFeatures-1])
    X = trainData[:,0:numberOfFeatures-1]
    X11 = X[:, 0] * X[:,0]
    X12 = X[:, 0] * X[:,1]
    X = numpy.hstack((X, X11.reshape((X.shape[0],1)), X12.reshape((X.shape[0],1))))

    Xscaled = scaleFeature(X)
    startOfLinearCaseWithHigherOrder = time.time()
    #Have kept the regularized parameter there just so that it can be played with.
    #Numbers mentioned below are without usage of regularized parameters
    # higher order linear case is 10X faster with L2 of 0.1 and requires lesser iteration than linear case also
    logisticRegression = LogisticRegression(Xscaled, Y, 0.01, 0.0001, 2000, 0, 0,verbose=False)
    logisticRegression.runGradientAscent()
    endOfLinearCaseWithHigherOrder = time.time()
    params = unscaleWeights(logisticRegression.weights, X)
    print("Iterations required For linear Case with higher order = " + str(logisticRegression.iteration))
    print("Final Cost for linear case with higher order " + str(logisticRegression.getLogLikelihood()))
    print("Accuracy of linear Classifier with higher order = " + str(getAccuracy(params, testData,order = 2)))
    print("Time taken for linear Classifier with higher order = " + str(endOfLinearCaseWithHigherOrder - startOfLinearCaseWithHigherOrder))
    plotDecisionBoundary(dataSet, params, order=2)
    
if __name__ == "__main__":
    dataSet = loadtxt('data/ex2data1.txt',delimiter = ",")
    trainData, testData = partitionTrainAndTest(dataSet, testRatio=0.2, shouldShuffle=False)
    showLinearExample(trainData, testData, dataSet)
    showLinearExampleWithHigherOrder(trainData, testData, dataSet)
    show()
    
    
    # ----------------------Sample Output-----------------------#
    #         Iterations required for linear case = 483         #
    #         Final Cost for linear case= 14.6568483082         #
    #         Accuracy of linear Classifier = 85.7142857143     #
    #     Time taken for linear Classifier = 0.0870389938354    #
    # Iterations required For linear Case-higher order = 2000   #
    #   Final Cost for linear Case-higher order= 4.6756902117   #
    #    Accuracy of linear Classifier higher order = 100.0     #
    # Time taken linear Classifier higher order = 0.41979789733 #
    # ----------------------------------------------------------#

    