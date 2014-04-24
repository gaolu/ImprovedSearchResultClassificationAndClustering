import sys
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from dataProcessor import *
from classifiers import *
class queryClassifier:
    def classifyQuery(self, opts, dataTrain, labelTrain, vectorizer, featureNames, categories):
        while True:
            prompt = 'Please input your query, or input "end" to end the program: '
            queryString = self.getInput(prompt)
            if queryString == 'end':
                sys.exit(0)
            # print queryString
            vectorQuery = vectorizer.transform([queryString])
            # print type(queryString), type(vectorQuery)
            # print vectorQuery.shape
            
            if opts.select_chi2:
                queryDataProcessor = dataProcessor()
                dataTrain, vectorQuery = queryDataProcessor.chiFeatueSel(opts, dataTrain, labelTrain, vectorQuery)
            
            queryClassifiers = classifiers()
            queryResults = queryClassifiers.queryClassifier(opts, MultinomialNB(alpha=0.01), dataTrain, labelTrain, vectorQuery, featureNames, categories)
        return
    
    def getInput(self, prompt):
        while True:
            s = raw_input(prompt)
            return s