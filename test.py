import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import collections
import types            #Deprecated

from DecisionTree import DecisionTree

#Test / Geometric Detection
#Let's create some test data: defined on a 3x3 square comprising (0,3) and (0,3)
n = 1000
x = np.random.random_sample((n,2))*3
#Let's create a plus sign on top of it on range x == (1,2), y == (1,2)
y = np.reshape(
        np.logical_or(
            np.logical_and(x[:,0]>1,x[:,0]<2),
            np.logical_and(x[:,1]>1,x[:,1]<2)),
    (1,n))

dataSet = np.hstack((x,y.T))
dataLength = len(dataSet)

print(dataSet)

plt.scatter(x[:,0],x[:,1],c=y)
plt.show()


DT = DecisionTree()

for i in range(0,2):
    #print DT.findMinimumSplit(dataSet[np.random.choice(range(0,dataLength),size=dataLength*.5,replace=False)],1,2,0,1)
    print DT.chooseAttribute(dataSet[np.random.choice(range(0,dataLength),size=dataLength*.5,replace=False)], [0,0,1], 2)



#################   FINISHED TESTS

#Test / findMinimumSplit for Continuous->Discrete
#n = 50
#yValues = [0, 1, 2]
#yProb = .7
#yProbs1 = [1, 0, 0]
#yProbs2 = [0, 0, 1]
#xValues = np.random.random(n*.7)*10+10
#x = np.concatenate((np.random.random(n*.3*2)*10+10,
#                    np.random.random(n*.7*2)*100+20))
#y = np.concatenate((np.random.choice(yValues, size=n, p=yProbs1),
#                    np.random.choice(yValues, size=n, p=yProbs2)))
#
#DS = DecisionStump()
#dataSet = np.vstack((x, y)).T
#DS.findMinimumSplit(dataSet,0,1,0,1,'g')

#Test / findMinimumSplit for Continuous->Continuous / Test 1
#l = []
#for i in range(0,100):
#    n = 50
#    y = np.arange(0,n*2)
#    x = np.concatenate((np.random.random(n*.8*2)*2+10,
#                        np.random.random(n*.2*2)*2+11))
#    
#    DS = DecisionStump()
#    dataSet = np.vstack((x, y)).T
#    splitIndex = DS.findMinimumSplit(dataSet,0,1,0,0)
#    l.append(splitIndex)
#    
#print l
#print np.mean(l)
#print np.std(l)

#Test / findMinimumSplit for Continuous->Continuous / Test 2
#l = []
#for i in range(0,1000):
#    n = 50
#    
#    x = np.concatenate((np.random.random(n*.8*2)*10+10,
#                        np.random.random(n*.2*2)*10+20))
#    y = np.concatenate((np.random.random(n*.8*2)*2+10,
#                        np.random.random(n*.2*2)*2+20))
#    
#    
#    DS = DecisionStump()
#    dataSet = np.vstack((x, y)).T
#    splitIndex = DS.findMinimumSplit(dataSet,0,1,0,0)
#    l.append(splitIndex)
#    
#print l
#print np.mean(l)
#print np.std(l)

#Test / findMinimumSplit for Discrete->Discrete
#n = 50
#yValues = [0, 1, 2]
#xValues = ['a', 'b', 'c']
#xProbs1 = [1, 0, 0]
#yProbs1 = [.3,.3,.4]
#yProbs2 = [0, 1, 0]
#xProbs2 = [0, 1, 0]
#y = np.concatenate((np.random.choice(yValues, size=n, p=yProbs1),
#                    np.random.choice(yValues, size=n, p=yProbs2)))
#x = np.concatenate((np.random.choice(xValues, size=n, p=xProbs1),
#                    np.random.choice(xValues, size=n, p=xProbs2)))
#
#DT = DecisionTree()
#dataSet = np.vstack((x, y)).T
#print dataSet
#print DT.findMinimumSplit(dataSet,0,1,1,1,'i')

#Test / findMinimumSplit for Discrete -> Continuous
#n = 50
#xValues = [0, 1, 2]
#xProbs1 = [0, 0, 1]
#xProbs2 = [.5, .5, 0]
#l = []
#for i in range(0,20):
#    x = np.reshape(
#            np.concatenate((np.random.choice(xValues, size=n*.7*2, p=xProbs1),
#                            np.random.choice(xValues, size=n*.3*2, p=xProbs2))),
#        (1,n*2))
#    y = np.reshape(
#            np.concatenate((np.random.random(2*n*.7)*2+10,
#                            np.random.random(2*n*.3)*2+20)),
#        (1,n*2))
#    
#    DT = DecisionTree()
#    dataSet = np.vstack((x,y)).T
#    splitIndex = DT.findMinimumSplit(dataSet,0,1,1,0,'g').decisionPoint
#    l.append(splitIndex)
#    
#print l
#print collections.Counter(l).most_common()

#Test / findInformationGain
#n = 50
#xValues = ['a', 'b', 'c']
#xProbs1 = [1,0,0]
#xProbs2 = [0,0,1]
#x = np.concatenate((np.random.choice(xValues, size=n, p=xProbs1),
#                    np.random.choice(xValues, size=n, p=xProbs2)))
#DS = DecisionStump()
#print DS.findInformationGain(x)

#Test / findGiniImpurity
#n = 50
#xValues = ['a', 'b', 'c']
#xProbs1 = [.33, .33, .34]
#xProbs2 = [.33, .33, .34]
#x = np.concatenate((np.random.choice(xValues, size=n, p=xProbs1),
#                    np.random.choice(xValues, size=n, p=xProbs2)))
#DS = DecisionStump()
#print DS.findGiniImpurity(x)
