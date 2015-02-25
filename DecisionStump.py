#TODO: Create Map of already-determined points (i.e. if chooseAttributes has already done work, don't repeat it)
import numpy as np
import operator as op

from collections import Counter
from math import log
from scipy.stats import mode

class DecisionStump:
    def __init__(self):
        pass
    
    def constructTree(self):
        pass
    
    def findMinimumSplit(self, inDataSet, predictorIndex, regressorIndex, predictorType, regressorType, impurityType = 'g'):
        dataLength = len(inDataSet)
        
        if impurityType == 'g':
            impFun = self.findGiniImpurity
        elif impurityType == 'i':
            impFun = self.findInformationGain
        else:
            raise NameError('impurityType not recognized')
        
        if predictorType != 0 and predictorType != 1:
            raise NameError('predictorType not recognized')
        if regressorType != 0 and regressorType != 1:
            raise NameError('regressorType not recognized')
        
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
                lessVal = np.mean(regressorData[0:varIndex])
                moreVal = np.mean(regressorData[varIndex:dataLength])
                return  LeafNode().\
                        predictorIndex(predictorIndex).\
                        decisionPoint(sortedData[varIndex,predictorIndex]).\
                        logicalOperator(op.lt).\
                        trueVal(lessVal).\
                        falseVal(moreVal)
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
                    var = impFun(regressorData[i:dataLength])*(dataLength-i) + impFun(regressorData[0:i])*(i)
                    if var < minVar:
                        minVar = var
                        varIndex = i
                less = mode(regressorData[0:varIndex])
                lessPurity = less[1][0]/(varIndex-1)
                lessVal = less[0][0]
                more = mode(regressorData[varIndex:dataLength])
                morePurity = more[1][0]/(dataLength-varIndex)
                moreVal = more[0][0]
                if lessPurity <= morePurity:
                    return  LeafNode().\
                            predictorIndex(predictorIndex).\
                            decisionPoint(sortedData[varIndex,predictorIndex]).\
                            logicalOperator(op.lt).\
                            trueVal(lessVal).\
                            falseVal(moreVal)
                else:
                    return  LeafNode().\
                            predictorIndex(predictorIndex).\
                            decisionPoint(sortedData[varIndex,predictorIndex]).\
                            logicalOperator(op.gt).\
                            trueVal(moreVal).\
                            falseVal(lessVal)
        elif predictorType == 1: 
            if regressorType == 0: # Discrete predicts Continuous
                dct = dict(Counter(inDataSet[:,predictorIndex]).most_common())
                predictorValues = dct.keys()
                #Find split point
                minVar = float("inf")
                varIndex = None
                for pval in predictorValues:
                    f = lambda k: inDataSet[:,predictorIndex] == pval
                    subData = inDataSet[f(inDataSet)]
                    var = np.std(subData[:,regressorIndex])
                    if var < minVar:
                        minVar = var
                        varIndex = pval
                f = lambda k: inDataSet[:,predictorIndex] == varIndex
                trueVal = np.mean(inDataSet[f(inDataSet)])
                fnot = lambda k: inDataSet[:,predictorIndex] != varIndex
                falseVal = np.mean(inDataSet[fnot(inDataSet)])
                return  LeafNode().\
                        predictorIndex(predictorIndex).\
                        decisionPoint(varIndex).\
                        logicalOperator(op.eq).\
                        trueVal(trueVal).\
                        falseVal(falseVal)
            elif regressorType == 1: # Discrete predicts Discrete
                #Find available types
                dct = dict(Counter(inDataSet[:,predictorIndex]).most_common())
                predictorValues = dct.keys()
                #Find split point
                minVar = float("inf")
                varIndex = None
                for pval in predictorValues:
                    f = lambda k: inDataSet[:,predictorIndex] == pval
                    subData = inDataSet[f(inDataSet)]
                    var = impFun(subData[:,regressorIndex])*len(subData)
                    if var < minVar:
                        minVar = var
                        varIndex = pval
                f = lambda k: inDataSet[:,predictorIndex] == varIndex
                trueVal = mode(inDataSet[f(inDataSet)])[0][0]
                fnot = lambda k: inDataSet[:,predictorIndex] != varIndex
                falseVal = mode(inDataSet[fnot(inDataSet)])[0][0]
                return  LeafNode().\
                        predictorIndex(predictorIndex).\
                        decisionPoint(varIndex).\
                        logicalOperator(op.eq).\
                        trueVal(trueVal).\
                        falseVal(falseVal)

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

class LeafNode:
    def __init__(self):
        pass
    
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
    
    def trueValProportions(self, trueValProportions):
        self.trueValProportions = trueValProportions
        return self
    
    def falseValProportions(self, falseValProportions):
        self.falseValProportions = falseValProportions
        return self

    def __str__(self):
        return  "IF data[" + str(self.predictorIndex) + "] " + \
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


        
    #DEPRECATED
    #def findGiniImpurity(possibleValues, totalSet):
    #    setLength = len(totalSet)
    #    giniImpurity = 1
    #    if type(x).__name__ == 'ndarray':
    #        for value in possibleValues:
    #            giniImpurity -= (float(totalSet.tolist().count(value))/setLength)**2
    #        return giniImpurity
    #    elif type(x).__name__ == 'list':
    #        for value in possibleValues:
    #            giniImpurity -= (float(totalSet.count(value))/setLength)**2
    #        return giniImpurity
        
        
    #def __findProportion(self, possibleValues, totalSet):
    #    setLength = len(totalSet)
    #    proportions = dict.fromkeys(possibleValues)
    #    for value in possibleValues:
    #        proportions[value] = totalSet.count(value)/setLength
    #    return proportions
    
    #def findMinimumSplit(self, inDataSet, predictorIndex, regressorIndex, impurityType = 'g'):
    #    dataLength = len(inDataSet)
    #    sortedData = inDataSet[np.array(inDataSet[:,predictorIndex].argsort())]
    #    predictorData = sortedData[:,predictorIndex]
    #    regressorData = sortedData[:,regressorIndex]
    #    
    #    if impurityType == 'r': #Continuous (Variance Reduction)
    #        minVar = float("inf")
    #        varIndex = -1
    #        for i in range(1,self.dataLength-1):
    #            splitVar = np.std(regressorData[0:i])+np.std(regressorData[i:dataLength])
    #            if splitVar < minVar:
    #                minVar = splitVar
    #                varIndex = i
    #            #end if
    #        #end for
    #        raise NotImplementedError('Finish this!')
    #    elif impurityType == 'i': #Information Gain
    #        dct = dict(Counter(regressorData).most_common())
    #        values = dct.keys()
    #        minVar = float("inf")
    #        varIndex = -1
    #        for i in range(1,(dataLength-1)):
    #            #var = self.findGiniImpurity(values, regressorData[i:dataLength])+self.findGiniImpurity(values, regressorData[0:i]) #DEPRECATED
    #            var = self.findInformationGain(regressorData[i:dataLength]) + self.findInformationGain(regressorData[0:i])
    #            if var < minVar:
    #                minVar = var
    #                varIndex = i
    #            #end if
    #            print str(i)+": "+str(var)
    #        #end for
    #        print varIndex  #TEST
    #        print minVar    #TEST
    #        
    #    elif impurityType == 'g': #Gini Impurity
    #        dct = dict(Counter(regressorData).most_common())
    #        values = dct.keys()
    #        minVar = float("inf")
    #        varIndex = -1
    #        for i in range(1,(dataLength-1)):
    #            var = self.findGiniImpurity(regressorData[i:dataLength]) + self.findGiniImpurity(regressorData[0:i])
    #            if var < minVar:
    #                minVar = var
    #                varIndex = i
    #            #end if
    #        #end for
    #        print varIndex  #TEST
    #        print minVar    #TEST
    #    else:
    #        raise NameError('Variance Type not defined')
    #    #end if
    ##end def

#class LeafNode:
#    def __init__(self, predictorIndex, decisionPoint, logicalOperator, trueVal, falseVal):
#        self.predictorIndex = predictorIndex
#        self.logicalOperator = logicalOperator
#        self.decisionPoint = decisionPoint
#        self.trueVal = trueVal
#        self.falseVal = falseVal
#    
#    #def __init__(self, predictorIndex, decisionPoint, logicalOperator, trueVal, falseVal, trueValProportions, falseValProportions):
#    #    self.predictorIndex = predictorIndex
#    #    self.logicalOperator = logicalOperator
#    #    self.decisionPoint = point
#    #    self.trueVal = trueVal
#    #    self.falseVal = falseVal
#    #    self.trueValProportions = trueValProportions
#    #    self.falseValProportions = falseValProportions
#
#    def __str__(self):
#        return "IF data[" + str(self.predictorIndex) + "] " + self.__stringifyOperator(self.logicalOperator) + str(self.decisionPoint) + ", THEN " + str(self.trueVal) + ", ELSE " + str(self.falseVal)
#        
#    def __stringifyOperator(self,logicalOperator):
#        if logicalOperator == op.lt:
#            return '<'
#        elif logicalOperator == op.gt:
#            return '>'
#        elif logicalOperator == op.eq:
#            return '=='
#        else:
#            raise NotImplementedError()
