from sysConfig import *
from dataLoader import *
from argParse import *
from dataProcessor import *
from classifiers import *
from resultPlot import *
from queryClassifier import *
from optparse import OptionParser
import numpy as np

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
    
    myDataLoader.printFileSize(trainData, trainDataSize, testData, testDataSize, categories)
    
    labelTrain = trainData.target
    labelTest = testData.target
    
    myDataProcessor = dataProcessor()
    
    dataTrain, vectorizer = myDataProcessor.trainFeatureExtract(opts, trainData, trainDataSize)
    
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
    
    myQueryClassifier = queryClassifier()
    myQueryClassifier.classifyQuery(opts, dataTrain, labelTrain, vectorizer, featureNames, categories)
    
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