import numpy as np
import matplotlib.pyplot as plt

import collections

from DecisionTreeNode import BranchNode

#Test / Geometric Detection
def test_GeometricCorner(display = 0):
	#Create Test Data: defined on a 3x3 square comprising (0,3) and (0,3)
	n = 1000
	x = np.random.random_sample((n,2))*3
	#Let's create a dividing line on x = 1
	y = np.reshape(
			np.logical_and(x[:,0] > 2,x[:,1] > 1),
		(1,n))

	dataSet = np.hstack((x,y.T))
	dataLength = len(dataSet)

	if display:
		plt.scatter(x[:,0],x[:,1],c=y)
		plt.show()

	DT = 	BranchNode().\
			set_maxDepth(2).\
			set_regressorIndex(2).\
			set_columnTypes([0,0,1])

	for i in range(0,1):
	    #print DT.findMinimumSplit(dataSet[np.random.choice(range(0,dataLength),size=dataLength*.5,replace=False)],1,2,0,1)
	    DT = DT.set_dataSet(dataSet[np.random.choice(range(0,dataLength),size=dataLength*.5,replace=False)])
	    print DT.constructTreeFromNode()


#Test / Geometric Detection
def test_GeometricCross(display = 0):
	#Create Test Data: defined on a 3x3 square comprising (0,3) and (0,3)
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

	if display:
		plt.scatter(x[:,0],x[:,1],c=y)
		plt.show()

	DT = 	BranchNode().\
			set_maxDepth(3).\
			set_regressorIndex(2).\
			set_columnTypes([0,0,1])

	for i in range(0,1):
	    #print DT.findMinimumSplit(dataSet[np.random.choice(range(0,dataLength),size=dataLength*.5,replace=False)],1,2,0,1)
	    DT = DT.set_dataSet(dataSet[np.random.choice(range(0,dataLength),size=dataLength*1,replace=False)])
	    print DT.constructTreeFromNode()

# Test / findMinimumSplit for Continuous->Discrete
def test_FindMinimumSplit_ContinuousDiscrete():
	n = 50
	yValues = [0, 1, 2]
	yProbs1 = [.5, .5, 0]
	yProbs2 = [0, .1, .9]
	x = np.concatenate((np.random.random(n)*10+10,		#average 15 (max 20)
	                   np.random.random(n)*10+20))		#average 25 (min 20)
	y = np.concatenate((np.random.choice(yValues, size=n, p=yProbs1),
	                   np.random.choice(yValues, size=n, p=yProbs2)))

	DT = BranchNode()
	DT = DT.set_depth(5)
	DT = DT.set_impurityType('g')
	dataSet = np.vstack((x, y)).T
	DT = DT.findMinimumSplit(dataSet,0,1,0,1)
	# DT.establishProportions()
	print DT
	decisionPoint = DT.decisionPoint
	if decisionPoint < 21 and decisionPoint > 20:
		print '*\t\tContinuous Discrete passed'
	else:
		print '**\t\tContinuous Discrete failed'

#Test / findMinimumSplit for Continuous->Continuous / Test 1
def test_FindMinimumSplit_ContinuousContinuous_1():
	l = []
	for i in range(0,100):
	   n = 100
	   y = np.arange(0,n)
	   x = np.concatenate((np.random.random(n*.5)*2+10,		#average 11 (max 12)
	                       np.random.random(n*.5)*2+12))	#average 13 (min 12)
	   
	   DT = BranchNode()
	   dataSet = np.vstack((x, y)).T
	   DT = DT.findMinimumSplit(dataSet,0,1,0,0)
	   l.append(DT.decisionPoint)
	   
	print DT
	decisionPoint = np.mean(l)
	if decisionPoint < 12.5 and decisionPoint > 11.5:
		print '*\t\tContinuous Continuous Test 1 passed'
	else:
		print '**\t\tContinuous Continuous Test 1 failed'
	print np.mean(l)
	print np.std(l)



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
def test_FindMinimumSplit_DiscreteDiscrete():
	n = 50
	yValues = [0, 1, 2]
	xValues = ['a', 'b', 'c']
	xProbs1 = [1, 0, 0]
	yProbs1 = [.3,.3,.4]
	yProbs2 = [0, 1, 0]
	xProbs2 = [0, 1, 0]
	y = np.concatenate((np.random.choice(yValues, size=n, p=yProbs1),
	                   np.random.choice(yValues, size=n, p=yProbs2)))
	x = np.concatenate((np.random.choice(xValues, size=n, p=xProbs1),
	                   np.random.choice(xValues, size=n, p=xProbs2)))

	DT = BranchNode()
	dataSet = np.vstack((x, y)).T
	DT = DT.findMinimumSplit(dataSet,0,1,1,1)
	if DT.trueVal == '1' and DT.decisionPoint == 'b' and DT.trueValProportions == 1.0:
		print '*\t\tDiscrete Discrete Test passed'
	else:
		print '**\t\tDiscrete Discrete Test failed'

#Test / findMinimumSplit for Discrete -> Continuous
def test_FindMinimumSplit_DiscreteContinuous():
	n = 100
	xValues = [0, 1, 2]
	xProbs1 = [0, 0, 1]
	xProbs2 = [.5, .5, 0]
	l = []
	for i in range(0,20):
	   x = np.reshape(
	           np.concatenate((np.random.choice(xValues, size=n*.5, p=xProbs1),
	                           np.random.choice(xValues, size=n*.5, p=xProbs2))),
	       (1,n))
	   y = np.reshape(
	           np.concatenate((np.random.random(n*.5)*.5+10,	#Average 10.25 (range 0.5) <- lower variance should be taken
	                           np.random.random(n*.5)*2+20)),	#Average 21.00 (range 2)
	       (1,n))
	   
	   DT = BranchNode()
	   dataSet = np.vstack((x,y)).T
	   DT = DT.findMinimumSplit(dataSet,0,1,1,0)
	   splitIndex = DT.decisionPoint
	   l.append(splitIndex)
	   
	print DT
	if collections.Counter(l).most_common()[0][0] == 2.0:
		print '*\t\tDiscrete Continuous Test passed'
	else:
		print '**\t\tDiscrete Continuous Test failed'

#Test / findInformationGain
def test_InformationGain():
	n = 100
	DT = BranchNode().set_impurityType('i')

	data = np.arange(1,n) 
	maxIG = DT.impurityFunction(data)

	data = np.ones(n)
	minIG = DT.impurityFunction(data)

	data = np.concatenate((np.zeros(n-1),np.ones(1)))
	midIG = DT.impurityFunction(data)

	if minIG == 0 and (minIG < midIG < maxIG):
		print '*\t\tInformation Gain Test passed'
	else:
		print '**\t\tInformation Gain Test failed'

#Test / findGiniImpurity
def test_GiniImpurity():
	n = 100
	DT = BranchNode().set_impurityType('i')

	data = np.arange(1,n) 
	maxIG = DT.impurityFunction(data)

	data = np.ones(n)
	minIG = DT.impurityFunction(data)

	data = np.concatenate((np.zeros(n-1),np.ones(1)))
	midIG = DT.impurityFunction(data)

	if minIG == 0 and (minIG < midIG < maxIG):
		print '*\t\tGini Impurity Test passed'
	else:
		print '**\t\tGini Impurity Test failed'

if __name__ == '__main__':
	constantSeed = 1
	if constantSeed:
		np.random.seed(seed=17)
	testIG = 0
	testGI = 0
	testCD = 0
	testCC = 0
	testDD = 0
	testDC = 0
	testGeometricCorner = 0
	testGeometricCross = 1
	display = 1
	if testIG:
		test_InformationGain()
	if testGI:
		test_GiniImpurity()
	if testCD:
		test_FindMinimumSplit_ContinuousDiscrete()
	if testCC:
		test_FindMinimumSplit_ContinuousContinuous_1()
	if testDD:
		test_FindMinimumSplit_DiscreteDiscrete()
	if testDC:
		test_FindMinimumSplit_DiscreteContinuous()
	if testGeometricCorner:
		test_GeometricCorner(display = display)
	if testGeometricCross:
		test_GeometricCross(display = display)
