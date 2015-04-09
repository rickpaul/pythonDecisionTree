#TODO: Create HashMap of already-determined points (i.e. if chooseAttributes has already done work, don't repeat it)
import numpy as np
import operator as op
from copy import copy
from math import log
from collections import Counter
from scipy.stats import mode
    
class Node:
    def __init__(self, depth = 0, impurityType = 'g', hasChildren = False):
        self.set_depth(depth) #Default Value
        self.set_impurityType(impurityType) #Default Value
        self.set_hasChildren(hasChildren) #Default Value
    
    def set_impurityType(self, impurityType):
        if impurityType != 'g' and impurityType != 'i':
            raise NameError('impurityType not recognized')
        self.impurityType = impurityType
        return self

    def set_dataSet(self, dataSet):
        self.dataSet = dataSet
        return self

    def set_columnTypes(self, columnTypes):
        self.columnTypes = columnTypes
        return self

    def set_predictorIndex(self, predictorIndex):
        self.predictorIndex = predictorIndex
        return self

    def set_regressorIndex(self, regressorIndex):
        self.regressorIndex = regressorIndex
        return self

    def set_hasChildren(self, hasChildren): # Deprecated, probably
        self.hasChildren = hasChildren
        return self

    def set_maxDepth(self, maxDepth):
        self.maxDepth = maxDepth
        return self

    def set_depth(self, depth):
        self.depth = depth
        return self

    def set_impurityValue(self, impurityValue):
        self.impurityValue = impurityValue
        return self

    #################### Tree Construction Methods / Discrete Impurity

    def impurityFunction(self, totalSet):
        if self.impurityType == 'g':
            return self.findGiniImpurity(totalSet)
        elif self.impurityType == 'i':
            return self.findInformationGain(totalSet)
        else:
            raise NameError('impurityType not recognized')
            # Necessary? Maybe not strictly...

    #Bigger values are worse; 0 is best
    def findGiniImpurity(self, totalSet):
        setLength = len(totalSet)
        giniImpurity = 1
        dct = dict(Counter(totalSet).most_common())
        for num in dct.values():
            giniImpurity -= (float(num)/setLength)**2
        return giniImpurity
    
    #Bigger values are worse; 0 is best
    def findInformationGain(self, totalSet):
        setLength = len(totalSet)
        informationGain = 0
        dct = dict(Counter(totalSet).most_common())
        for num in dct.values():
            freq = (float(num)/setLength)
            informationGain -= freq*log(freq,2)
        return informationGain

class BranchNode(Node):
    def __init__(self):
        Node.__init__(self)
    
    def set_trueDataSet(self, trueDataSet):
        self.trueDataSet = trueDataSet
        return self
    
    def set_logicalOperator(self, logicalOperator):
        self.logicalOperator = logicalOperator
        return self

    def set_decisionPoint(self, decisionPoint):
        self.decisionPoint = decisionPoint
        return self

    def set_falseDataSet(self, falseDataSet):
        self.falseDataSet = falseDataSet
        return self
    
    def set_trueChildNode(self, trueChildNode):
        self.trueChildNode = trueChildNode
        return self
    
    def set_falseChildNode(self, falseChildNode):
        self.falseChildNode = falseChildNode
        return self

    def set_trueVal(self, trueVal): # Deprecated, probably (or only for curiosity)
        self.trueVal = trueVal
        return self

    def set_falseVal(self, falseVal): # Deprecated, probably (or only for curiosity)
        self.falseVal = falseVal
        return self
    
    def set_trueValProportions(self, trueValProportions): # Deprecated, probably (or only for curiosity)
       self.trueValProportions = trueValProportions
       return self
    
    def set_falseValProportions(self, falseValProportions): # Deprecated, probably (or only for curiosity)
       self.falseValProportions = falseValProportions
       return self

    #################### Tree Construction Methods
    def constructTreeFromNode(self):
        if self.evaluateStoppingConditions():
            leaf =  LeafNode().\
                    set_depth(self.depth).\
                    set_maxDepth(self.maxDepth).\
                    set_dataSet(self.dataSet).\
                    set_regressorIndex(self.regressorIndex).\
                    set_columnTypes(self.columnTypes).\
                    set_impurityType(self.impurityType)
            leaf.populateCharacteristics()
            return leaf
        else:
            self = self.chooseAttribute()
            self.set_hasChildren(True)
            self.set_trueChildNode( BranchNode().\
                                    set_depth(self.depth+1).\
                                    set_maxDepth(self.maxDepth).\
                                    set_dataSet(self.trueDataSet).\
                                    set_regressorIndex(self.regressorIndex).\
                                    set_columnTypes(self.columnTypes)
                                    )
            self.set_falseChildNode(BranchNode().\
                                    set_depth(self.depth+1).\
                                    set_maxDepth(self.maxDepth).\
                                    set_dataSet(self.falseDataSet).\
                                    set_regressorIndex(self.regressorIndex).\
                                    set_columnTypes(self.columnTypes)
                                    )
            self.set_falseChildNode(self.falseChildNode.constructTreeFromNode())
            self.set_trueChildNode(self.trueChildNode.constructTreeFromNode())
            return self

    def evaluateStoppingConditions(self):
        # Check if Max Depth Reached
        if self.depth > self.maxDepth:
            print 'Stopping because of depth' #TEST
            return True
        # Check if Data Set Size is Sufficient
        if len(self.dataSet) <= 1:
            return True
        # Check if Minimal Impurity Reached
        #CONSIDER: Basing the minimal impurity on a threshhold
        regressorType = self.columnTypes[self.regressorIndex]
        if regressorType == 1:
            if self.impurityFunction(self.dataSet[:,self.regressorIndex]) == 0.0:
                print 'Stopping because of impurity' #TEST
                return True
        elif regressorType == 0:
            if np.std(self.dataSet[:,self.regressorIndex]) == 0.0:
                print 'Stopping because of impurity' #TEST
                return True
        return False

    def chooseAttribute(self):
        dataLength = len(self.dataSet)
        numCols = len(self.dataSet[0,:])
        regressorType = self.columnTypes[self.regressorIndex]
        
        if regressorType == 0: #Continuous
            baseLine = np.std(self.dataSet[:,self.regressorIndex]) * dataLength
        elif regressorType == 1: #Discrete
            baseLine = self.impurityFunction(self.dataSet[:,self.regressorIndex]) * dataLength
        else:
            raise NameError('regressorType not recognized')

        print "BASELINE: " + str(baseLine)  #TEST

        maxReduction = -1
        maxReductionNode = None

        for i in range(0, numCols):
            if i == self.regressorIndex:
                continue
            else:
                node = self.findMinimumSplit(self.dataSet, i, self.regressorIndex, self.columnTypes[i], regressorType)
                reduction = baseLine - node.impurityValue
                
                #print "  " + str(i) + " : " + str(reduction) + " : " + str(node) + " : " + str(len(node.trueDataSet)) + "/" + str(len(node.falseDataSet)) #TEST
                
                if reduction > maxReduction:
                    maxReduction = reduction
                    maxReductionNode = node
                    
        return maxReductionNode

    def findMinimumSplit(self, inDataSet, predictorIndex, regressorIndex, predictorType, regressorType):
        #establish dataLength
        dataLength = len(inDataSet)
        
        #Check for purity of inputs
        #(yes, this is being done above, too, but there are no big computational costs to doing it here as well)
        if predictorType != 0 and predictorType != 1:
            raise NameError('predictorType not recognized')
        if regressorType != 0 and regressorType != 1:
            raise NameError('regressorType not recognized')

        # Find Minimum Split
        if predictorType == 0:
            # Find Minimum Split / Continuous Predicts Continuous
            if regressorType == 0: 
                # Find Minimum Split / Continuous Predicts Continuous / Find Split Point
                sortedData = inDataSet[np.array(inDataSet[:,predictorIndex].argsort())]
                regressorData = sortedData[:,regressorIndex]
                minVar = float("inf")
                varIndex = -1
                for i in range(1,dataLength-1):
                    var = np.std(regressorData[0:i])*(i)+np.std(regressorData[i:dataLength])*(dataLength-i)
                    if var < minVar:
                        minVar = var
                        varIndex = i
                # Find Minimum Split / Continuous Predicts Continuous / Establish Data For Return
                lessImpurity = np.std(regressorData[0:varIndex])*(varIndex)
                moreImpurity = np.std(regressorData[varIndex:dataLength])*(dataLength-varIndex)
                lessData = sortedData[0:varIndex,:]
                lessVal = np.mean(lessData[:,regressorIndex])
                moreData = sortedData[varIndex:dataLength,:]
                moreVal = np.mean(moreData[:,regressorIndex])
                if lessImpurity < moreImpurity:
                    logicalOperator = op.lt
                    trueVal = lessVal
                    falseVal = moreVal
                    trueData = lessData
                    falseData = moreData
                else:
                    logicalOperator = op.gt
                    trueVal = moreVal
                    falseVal = lessVal
                    trueData = moreData
                    falseData = moreData
                decisionPoint = sortedData[varIndex,predictorIndex]
                impurityValue = minVar
                trueValProportions = None
                falseValProportions = None
            # Find Minimum Split / Continuous predicts Discrete
            elif regressorType == 1: 
                # Find Minimum Split / Continuous Predicts Discrete / Sort Data
                sortedData = inDataSet[np.array(inDataSet[:,predictorIndex].argsort())]
                predictorData = sortedData[:,predictorIndex]
                regressorData = sortedData[:,regressorIndex]
                # Find Minimum Split / Continuous Predicts Discrete / Find All Discrete Values
                dct = dict(Counter(regressorData).most_common())
                values = dct.keys()
                # Find Minimum Split / Continuous Predicts Discrete / Find Split Point
                minVar = float("inf")
                varIndex = -1
                for i in range(1,(dataLength-1)):
                    var = (self.impurityFunction(regressorData[i:dataLength])*(dataLength-i) +
                           self.impurityFunction(regressorData[0:i])*(i))
                    if i >= 332 and i <= 339: #TEST
                        junk = 1 #TEST
                    if var < minVar:
                        minVar = var
                        varIndex = i
                # Find Minimum Split / Continuous Predicts Discrete / Establish Data For Return
                lessData = sortedData[0:varIndex,:]
                lessPurity = mode(regressorData[0:varIndex])[1][0]/(varIndex)
                lessVal = mode(regressorData[0:varIndex])[0][0]
                moreData = sortedData[varIndex:dataLength,:]
                morePurity = mode(regressorData[varIndex:dataLength])[1][0]/(dataLength-varIndex)
                moreVal = mode(regressorData[varIndex:dataLength])[0][0]
                # if False: #TEST
                if lessPurity >= morePurity:
                    logicalOperator = op.lt
                    trueVal = lessVal
                    falseVal = moreVal
                    trueData = lessData
                    falseData = moreData
                    trueValProportions = lessPurity
                    falseValProportions = morePurity
                else:
                    logicalOperator = op.gt
                    trueVal = moreVal
                    falseVal = lessVal
                    trueData = moreData
                    falseData = lessData
                    trueValProportions = morePurity
                    falseValProportions = lessPurity
                decisionPoint = sortedData[varIndex,predictorIndex]
                impurityValue = minVar
        elif predictorType == 1: 
            # Find Minimum Split / Discrete predicts Continuous
            if regressorType == 0:
                # Find Minimum Split / Discrete predicts Continuous / Find Available Discrete Values
                dct = dict(Counter(inDataSet[:,predictorIndex]).most_common())
                predictorValues = dct.keys()
                if len(predictorValues) == 1: #CONSIDER: Removing this check once we've pushed it further up.
                    raise NotImplementedError('make it return a leaf node here.')
                # Find Minimum Split / Discrete predicts Continuous / Find Split Point
                minVar = float("inf")
                minVarValue = None
                subDataLen = None
                for pval in predictorValues:
                    f = lambda k: inDataSet[:,predictorIndex] == pval
                    subData = inDataSet[f(inDataSet),regressorIndex].astype(float)
                    var = np.std(subData)#*len(subData) #we don't multiply by datalength here because we don't want to penalize large datasets
                    if var < minVar:
                        subDataLen = len(subData) # used to multiply later
                        minVar = var
                        minVarValue = pval
                # Find Minimum Split / Discrete Predicts Continuous / Establish Data For Return
                f = lambda k: inDataSet[:,predictorIndex] == minVarValue
                fnot = lambda k: inDataSet[:,predictorIndex] != minVarValue
                logicalOperator = op.eq
                trueData = inDataSet[f(inDataSet),:]
                falseData = inDataSet[fnot(inDataSet),:]
                trueVal = np.mean(trueData[:,regressorIndex].astype(float))
                falseVal = np.mean(falseData[:,regressorIndex].astype(float))
                trueValProportions = None
                falseValProportions = None
                decisionPoint = minVarValue
                impurityValue = minVar*subDataLen#multiply to make comparisons equal between discrete and continuous predictors
            # Find Minimum Split / Discrete predicts Discrete
            elif regressorType == 1:
                # Find Minimum Split / Discrete predicts Discrete / Find Available Discrete Values
                dct = dict(Counter(inDataSet[:,predictorIndex]).most_common())
                predictorValues = dct.keys()
                if len(predictorValues) == 1: #CONSIDER: Removing this check once we've pushed it further up.
                    raise NotImplementedError('make it return a leaf node here.')
                # Find Minimum Split / Discrete predicts Discrete / Find Split Point
                minVar = float("inf")
                minVarValue = None
                for pval in predictorValues:
                    f = lambda k: inDataSet[:,predictorIndex] == pval
                    subData = inDataSet[f(inDataSet)]
                    var = self.impurityFunction(subData[:,regressorIndex])*len(subData)
                    if var < minVar:
                        minVar = var
                        minVarValue = pval
                # Find Minimum Split / Discrete predicts Discrete / Establish Data For Return
                f = lambda k: inDataSet[:,predictorIndex] == minVarValue
                fnot = lambda k: inDataSet[:,predictorIndex] != minVarValue
                logicalOperator = op.eq
                trueData = inDataSet[f(inDataSet)]
                falseData = inDataSet[fnot(inDataSet)]
                trueVal = mode(trueData[:,regressorIndex])[0][0]
                falseVal = mode(falseData[:,regressorIndex])[0][0]
                trueValProportions = mode(trueData[:,regressorIndex])[1][0]/len(trueData)
                falseValProportions = mode(falseData[:,regressorIndex])[1][0]/len(falseData)
                decisionPoint = minVarValue
                impurityValue = minVar
        return  copy(self).\
                set_predictorIndex(predictorIndex).\
                set_decisionPoint(decisionPoint).\
                set_logicalOperator(logicalOperator).\
                set_trueVal(trueVal).\
                set_falseVal(falseVal).\
                set_trueDataSet(trueData).\
                set_falseDataSet(falseData).\
                set_trueValProportions(trueValProportions).\
                set_falseValProportions(falseValProportions).\
                set_impurityValue(impurityValue)

    #################### To_String Methods
    
    def __str__(self):
        string =    ("    " * self.depth + 
                    "IF data[" + str(self.predictorIndex) + "] " + 
                    self.__stringifyOperator(self.logicalOperator) + " " + 
                    str(self.decisionPoint) + 
                    ", THEN " + str(self.trueVal) + ", ELSE " + str(self.falseVal))
        if self.trueValProportions is not None:
            string += " {True Certainty:" + str(self.trueValProportions) + " | False Certainty:" + str(self.falseValProportions) + "}"
        return (string + "\n" + 
                "    " * self.depth + "TRUE CHILD:" + "\n" + 
                str(self.trueChildNode) + "\n" +
                "    " * self.depth + "FALSE CHILD:" + "\n" + 
                str(self.falseChildNode))


        
    def __stringifyOperator(self,logicalOperator):
        if logicalOperator == op.lt:
            return '<'
        elif logicalOperator == op.gt:
            return '>'
        elif logicalOperator == op.eq:
            return '=='
        else:
            raise NameError('Logical Operator not recognized')
            

class LeafNode(Node):
    def __init__(self):
        Node.__init__(self)
        self.purityProportion = None

    def set_value(self, value):
        self.value = value
        return self

    def set_purityProportion(self, purityProportion):
        self.purityProportion = purityProportion
        return self

    def populateCharacteristics(self):
        regressorType = self.columnTypes[self.regressorIndex]
        # Continuous Regressor
        if regressorType == 0:
            self.set_value(np.mean())
            self.set_impurityValue(np.std(self.dataSet[:,self.regressorIndex]))
            self.set_purityProportion(None)
        elif regressorType == 1:
            self.set_value(mode(self.dataSet[:,self.regressorIndex])[0][0])
            self.set_impurityValue(self.impurityFunction(self.dataSet[:,self.regressorIndex]))
            self.set_purityProportion(mode(self.dataSet[:,self.regressorIndex])[1][0]/len(self.dataSet))
        else:
            raise NameError('regressorType not recognized')

    def __str__(self):
        string = (  "    " * self.depth + 
                    "Node Value: " + str(self.value) +
                    " Impurity: " + str(self.impurityValue) +
                    " NumPoints: " + str(len(self.dataSet)))
        if self.purityProportion is not None:
            string += " {True Certainty:" + str(self.purityProportion) + "}"
        return string
