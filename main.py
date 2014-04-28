from sysConfig import *
from dataLoader import *
from argParse import *
from dataProcessor import *
from classifiers import *
from resultPlot import *
from queryClassifier import *
from optparse import OptionParser
import numpy as np
import csv
from time import time
# temp
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

def main():
    # set log format
    myLogConfig = sysConfig()
    myLogConfig.logConfig()
    
    # parse command line parameters
    myArgParser = argParse()
    opts, args = myArgParser.parseArg()
    
    # load data
    myDataLoader = dataLoader()
    trainData, testData, categories = myDataLoader.loadData(opts)
    
    trainDataSize = myDataLoader.sizeMb(trainData.data)
    testDataSize = myDataLoader.sizeMb(testData.data)
    
    # print type(trainData), type(testData)
    
    myDataLoader.printFileSize(trainData, trainDataSize, testData, testDataSize, categories)
    
    labelTrain = trainData.target
    labelTest = testData.target
    
    myDataProcessor = dataProcessor()
    
    dataTrain, vectorizer = myDataProcessor.trainFeatureExtract(opts, trainData, trainDataSize)
    
    # print 'vectorizer is:', vectorizer
    
    # classCenter = myDataProcessor.getCenter(dataTrain, labelTrain)
    
    # f=open('classCenter.txt','w')
    # for i in classCenter.keys():
    #     f.write(repr(i))
    #     for n in classCenter[i]:
    #         f.write(' %s'%(repr(n)))
    #     f.write('\n')
    # f.close()
    print 'loading class centers...'
    time0 = time()
    classCenter = {}
    with open('classCenter.txt', 'r') as file:
        for line in file:
            elementList = line.rstrip().split()
            category = int(elementList[0])
            dataList = [float(element) for element in elementList[1 : ]]
            # print len(dataList)
            classCenter[category] = dataList
    timeSpan = time() - time0
    print 'done in', timeSpan, 'seconds'
    # dataTest = myDataProcessor.testFeatureExtract(opts, testData, testDataSize, vectorizer)
    
    # print dataTest.shape
    
    # if opts.select_chi2:
    #     dataTrain, dataTest = myDataProcessor.chiFeatueSel(opts, dataTrain, labelTrain, dataTest)
    
    # print 'before feature extraction'
    # print dataTrain.shape, vectorQuery.shape
    
    if opts.use_hashing:
        featureNames = None
    else:
        featureNames = np.asarray(vectorizer.get_feature_names())
    
    # dataQuery = ['which companie is better intel or amd']
    # print dataQuery
    # vectorQuery = vectorizer.transform(dataQuery)
    # print type(dataQuery), type(vectorQuery)
    # print vectorQuery.shape
        
    # if opts.select_chi2:
    #     dataTrain, vectorQuery = myDataProcessor.chiFeatueSel(opts, dataTrain, labelTrain, vectorQuery)
    
    queryFile = './queries.txt'
    queryList = myDataLoader.loadQuery(queryFile)
    
    myQueryClassifier = queryClassifier()
    # for firstN in range(10):
    firstN = 5
    myQueryClassifier.classifyQuery(opts, dataTrain, labelTrain, vectorizer, featureNames, categories, trainData, testData, queryList, classCenter, firstN)
    
    # queryFile = './queries.txt'
    
    sys.exit(1)
    
    # print 'after feature extraction'
    # print dataTrain.shape, vectorQuery.shape
    
    # myClassifiers = classifiers()
    # results = myClassifiers.getResults(opts, dataTrain, labelTrain, dataTest, labelTest, featureNames, categories)
    
    # myPlot = resultPlot()
    # myPlot.plots(results)
    
    
    # queryResults = myClassifiers.queryClassifier(opts, MultinomialNB(alpha=0.01), dataTrain, labelTrain, vectorQuery, featureNames, categories)
    
    

# start from main()
if __name__ == "__main__":
    main()