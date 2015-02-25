#TODO: Create HashMap of already-determined points (i.e. if chooseAttributes has already done work, don't repeat it)
import numpy as np
import operator as op

from collections import Counter
from math import log
from scipy.stats import mode

class DecisionTree:
    # def __init__(self, inDataSet, inColumnTypes, regressorIndex,  maxDepth = 2, impurityType = 'g'):
    def __init__(self):
        pass
    
    def constructTree(self, depth):
        pass
    
    def chooseAttribute(self, inDataSet, inColumnTypes, regressorIndex, impurityType = 'g'):
        dataLength = len(inDataSet)
        numCols = len(inDataSet[0,:])
        regressorType = inColumnTypes[regressorIndex]
        
        if regressorType == 0: #Continuous
            baseLine = np.std(inDataSet[:,regressorIndex]) * dataLength
        elif regressorType == 1: #Discrete
            baseLine = self.impurityFunction(inDataSet[:,regressorIndex],impurityType) * dataLength
        else:
            raise NameError('regressorType not recognized')

        print "BASELINE: " + str(baseLine)  #TEST

        maxReduction = -1
        maxReductionNode = None

        for i in range(0, numCols):
            if i == regressorIndex:
                continue
            else:
                node = self.findMinimumSplit(inDataSet, i, regressorIndex, inColumnTypes[i], regressorType, impurityType)
                reduction = baseLine - node.impurityValue
                
                print "  " + str(i) + " : " + str(reduction) + " : " + str(node) + " : " + str(len(node.trueDataSet)) + "/" + str(len(node.falseDataSet)) #TEST
                
                if reduction > maxReduction:
                    maxReduction = reduction
                    maxReductionNode = node
                    
        return maxReductionNode
    
    def findMinimumSplit(self, inDataSet, predictorIndex, regressorIndex, predictorType, regressorType, impurityType = 'g'):
        #establish dataLength
        dataLength = len(inDataSet)
        
        #Check for purity of inputs
        #(yes, this is being done above, too, but there are no big computational costs to doing it here as well)
        if impurityType != 'g' and impurityType != 'i':
            raise NameError('impurityType not recognized')
        
        if predictorType != 0 and predictorType != 1:
            raise NameError('predictorType not recognized')
        if regressorType != 0 and regressorType != 1:
            raise NameError('regressorType not recognized')
        
        #find minimum split
        if predictorType == 0:
            if regressorType == 0: #Continuous predicts Continuous
                sortedData = inDataSet[np.array(inDataSet[:,predictorIndex].argsort())]
                regressorData = sortedData[:,regressorIndex]
                minVar = float("inf")
                varIndex = -1
                for i in range(1,dataLength-1):
                    var = np.std(regressorData[0:i])*(i)+np.std(regressorData[i:dataLength])*(dataLength-i)
                    if var < minVar:
                        minVar = var
                        varIndex = i
                lessData = sortedData[0:varIndex,:]
                lessVal = np.mean(lessData[:,regressorIndex])
                moreData = sortedData[varIndex:dataLength,:]
                moreVal = np.mean(moreData[:,regressorIndex])
                return  Node().\
                        predictorIndex(predictorIndex).\
                        decisionPoint(sortedData[varIndex,predictorIndex]).\
                        logicalOperator(op.lt).\
                        trueVal(lessVal).\
                        falseVal(moreVal).\
                        trueDataSet(lessData).\
                        falseDataSet(moreData).\
                        impurityValue(minVar)
            elif regressorType == 1: #Continuous predicts Discrete
                #Get sorted data
                sortedData = inDataSet[np.array(inDataSet[:,predictorIndex].argsort())]
                predictorData = sortedData[:,predictorIndex]
                regressorData = sortedData[:,regressorIndex]
                #Find available types
                dct = dict(Counter(regressorData).most_common())
                values = dct.keys()
                #Find split point
                minVar = float("inf")
                varIndex = -1
                for i in range(1,(dataLength-1)):
                    var = (self.impurityFunction(regressorData[i:dataLength],impurityType)*(dataLength-i) +
                           self.impurityFunction(regressorData[0:i],impurityType)*(i))
                    if var < minVar:
                        minVar = var
                        varIndex = i
                lessData = sortedData[0:varIndex,:]
                lessPurity = mode(regressorData[0:varIndex])[1][0]/(varIndex-1)
                lessVal = mode(regressorData[0:varIndex])[0][0]
                moreData = sortedData[varIndex:dataLength,:]
                morePurity = mode(regressorData[varIndex:dataLength])[1][0]/(dataLength-varIndex)
                moreVal = mode(regressorData[varIndex:dataLength])[0][0]
                #print "\n++++++++++\n" + str(lessPurity) + " : " + str(morePurity) + "\n++++++++++\n" #TEST
                #print "\n++++++++++\n" + str(lessVal) + " : " + str(moreVal) + "\n++++++++++\n"  #TEST
                #print "\n++++++++++\n" + str(lessData[1:3,:]) + "\n : \n" + str(moreData[1:3,:]) + "\n++++++++++\n"  #TEST
                if lessPurity <= morePurity:
                    return  Node().\
                            predictorIndex(predictorIndex).\
                            decisionPoint(sortedData[varIndex,predictorIndex]).\
                            logicalOperator(op.lt).\
                            trueVal(lessVal).\
                            falseVal(moreVal).\
                            trueDataSet(lessData).\
                            falseDataSet(moreData).\
                            impurityValue(minVar)
                else:
                    return  Node().\
                            predictorIndex(predictorIndex).\
                            decisionPoint(sortedData[varIndex,predictorIndex]).\
                            logicalOperator(op.gt).\
                            trueVal(moreVal).\
                            falseVal(lessVal).\
                            trueDataSet(moreData).\
                            falseDataSet(lessData).\
                            impurityValue(minVar)
                        
        elif predictorType == 1: 
            if regressorType == 0: # Discrete predicts Continuous
                #This should probably be done by finding a comparison between baseline and w/datasets removed datapoints
                dct = dict(Counter(inDataSet[:,predictorIndex]).most_common())
                predictorValues = dct.keys()
                #Check if worth doing work
                if len(predictorValues) == 1:
                    return  Node().\
                            predictorIndex(predictorIndex).\
                            decisionPoint(predictorValues[0]).\
                            logicalOperator(op.eq).\
                            trueVal(np.mean(inDataSet)[0][0]).\
                            falseVal(None).\
                            impurityValue(np.std(inDataSet[:,regressorIndex])*dataLength) 
                #Find split point
                minVar = float("inf")
                varIndex = None
                subDataLen = None
                for pval in predictorValues:
                    f = lambda k: inDataSet[:,predictorIndex] == pval
                    subData = inDataSet[f(inDataSet)]
                    var = np.std(subData[:,regressorIndex])#*len(subData) #we don't multiply by datalength here because we don't want to penalize large datasets
                    if var < minVar:
                        subDataLen = len(subData)
                        minVar = var
                        varIndex = pval
                f = lambda k: inDataSet[:,predictorIndex] == varIndex
                trueData = inDataSet[f(inDataSet)]
                trueVal = np.mean(trueData)
                fnot = lambda k: inDataSet[:,predictorIndex] != varIndex
                falseData = inDataSet[fnot(inDataSet)]
                falseVal = np.mean(falseData)
                return  Node().\
                        predictorIndex(predictorIndex).\
                        decisionPoint(varIndex).\
                        logicalOperator(op.eq).\
                        trueVal(trueVal).\
                        falseVal(falseVal).\
                        trueDataSet(trueData).\
                        falseDataSet(falseData).\
                        impurityValue(minVar*subDataLen) #multiply by subDataLen to make comparisons equal between discrete and continuous predictors
            elif regressorType == 1: # Discrete predicts Discrete
                #Find available types
                dct = dict(Counter(inDataSet[:,predictorIndex]).most_common())
                predictorValues = dct.keys()
                #Check if worth doing work
                if len(predictorValues) == 1:
                    return  Node().\
                            predictorIndex(predictorIndex).\
                            decisionPoint(predictorValues[0]).\
                            logicalOperator(op.eq).\
                            trueVal(mode(inDataSet)[0][0]).\
                            falseVal(None).\
                            impurityValue(self.impurityFunction(inDataSet[:,regressorIndex],impurityType)*dataLength) 
                #Find split point
                minVar = float("inf")
                varIndex = None
                for pval in predictorValues:
                    f = lambda k: inDataSet[:,predictorIndex] == pval
                    subData = inDataSet[f(inDataSet)]
                    var = self.impurityFunction(subData[:,regressorIndex],impurityType)*len(subData)
                    if var < minVar:
                        minVar = var
                        varIndex = pval
                f = lambda k: inDataSet[:,predictorIndex] == varIndex
                trueData = inDataSet[f(inDataSet)]
                trueVal = mode(trueData[:,regressorIndex])[0][0]
                fnot = lambda k: inDataSet[:,predictorIndex] != varIndex
                falseData = inDataSet[fnot(inDataSet)]
                falseVal = mode(falseData[:,regressorIndex])[0][0]
                return  Node().\
                        predictorIndex(predictorIndex).\
                        decisionPoint(varIndex).\
                        logicalOperator(op.eq).\
                        trueVal(trueVal).\
                        falseVal(falseVal).\
                        trueDataSet(trueData).\
                        falseDataSet(falseData).\
                        impurityValue(minVar) 

    def impurityFunction(self, totalSet, impurityType):
        if impurityType == 'g':
            return self.findGiniImpurity(totalSet)
        elif impurityType == 'i':
            return self.findInformationGain(totalSet)
        else:
            raise NameError('impurityType not recognized')

    def findGiniImpurity(self, totalSet):
        setLength = len(totalSet)
        giniImpurity = 1
        dct = dict(Counter(totalSet).most_common())
        for num in dct.values():
            giniImpurity -= (float(num)/setLength)**2
        return giniImpurity
    
    def findInformationGain(self, totalSet):
        setLength = len(totalSet)
        informationGain = 0
        dct = dict(Counter(totalSet).most_common())
        for num in dct.values():
            freq = (float(num)/setLength)
            informationGain -= freq*log(freq,2)
        return informationGain

class Node:
    def __init__(self, depth = 0):
        self.depth = depth #Default Value
        pass
    
    def depth(self, depth):
        self.depth = depth
        return self
    
    def predictorIndex(self, predictorIndex):
        self.predictorIndex = predictorIndex
        return self
    
    def logicalOperator(self, logicalOperator):
        self.logicalOperator = logicalOperator
        return self

    def decisionPoint(self, decisionPoint):
        self.decisionPoint = decisionPoint
        return self
    
    def trueVal(self, trueVal):
        self.trueVal = trueVal
        return self
    
    def falseVal(self, falseVal):
        self.falseVal = falseVal
        return self
    
    def trueDataSet(self, trueDataSet):
        self.trueDataSet = trueDataSet
        return self
    
    def falseDataSet(self, falseDataSet):
        self.falseDataSet = falseDataSet
        return self
    
    def trueChildNode(self, trueChildNode):
        self.trueChildNode = trueChildNode
        return self
    
    def falseChildNode(self, falseChildNode):
        self.falseChildNode = falseChildNode
        return self
    
    #def trueValProportions(self, trueValProportions):
    #    self.trueValProportions = trueValProportions
    #    return self
    #
    #def falseValProportions(self, falseValProportions):
    #    self.falseValProportions = falseValProportions
    #    return self
    
    def impurityValue(self, impurityValue):
        self.impurityValue = impurityValue
        return self
    
    def __str__(self):
        return  "    " * self.depth + \
                "IF data[" + str(self.predictorIndex) + "] " + \
                self.__stringifyOperator(self.logicalOperator) + " " + \
                str(self.decisionPoint) + \
                ", THEN " + str(self.trueVal) + ", ELSE " + str(self.falseVal)
        
    def __stringifyOperator(self,logicalOperator):
        if logicalOperator == op.lt:
            return '<'
        elif logicalOperator == op.gt:
            return '>'
        elif logicalOperator == op.eq:
            return '=='
        else:
            raise NotImplementedError()
    
    