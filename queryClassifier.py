import sys
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from dataProcessor import *
from classifiers import *
from resultCluster import *
class queryClassifier:
    
    def classifyQuery(self, opts, dataTrain, labelTrain, vectorizer, featureNames, categories, trainData, testData, queryList, classCenter, firstN):
        while True:
            prompt = 'Please input your query, or input "end" to end the program: '
            queryString = self.getInput(prompt)
            if queryString == 'end':
                return
        # randScore = 0.0
        # timeSpan = 0.0
        # testNo = 0
        # hitRate = 0.0
        # avgafTime = 0.0 
        # avgafRI = 0.0
        # avgafMI = 0.0 
        # avghirTime = 0.0 
        # avghirRI = 0.0 
        # avghirMI = 0.0 
        # avgdbTime = 0.0 
        # avgdbRI = 0.0 
        # avgdbMI = 0.0 
        # avgkmTime = 0.0 
        # avgkmRI = 0.0
        # avgkmMI = 0.0
            nClusters = 20
            testNo = 0
            majHitRate = 0.0
            appHitRate = 0.0
        # for queryString in queryList:
            # print queryString
            vectorQuery = vectorizer.transform([queryString])
            # print type(queryString), type(vectorQuery)
            # print vectorQuery.shape
            
            if opts.select_chi2:
                queryDataProcessor = dataProcessor()
                dataTrain, vectorQuery = queryDataProcessor.chiFeatueSel(opts, dataTrain, labelTrain, vectorQuery)
            
            queryClassifiers = classifiers()
            # print '==============================================='
            # print '==============================================='
            # print '==============================================='
            # print 'the query to be predicted is:', queryString
            queryResults = queryClassifiers.queryClassifier(opts, MultinomialNB(alpha=0.01), dataTrain, labelTrain, vectorQuery, featureNames, categories)
            
            # begin the clustering process
            myResultCluster = resultCluster()
            # afTime, afRI, afMI, hirTime, hirRI, hirMI, dbTime, dbRI, dbMI, kmTime, kmRI, kmMI = myResultCluster.getClusters(queryString, trainData, testData, vectorizer, categories, queryResults, classCenter, nClusters)
            # retSum = afTime + afRI + afMI + hirTime + hirRI + hirMI + dbTime + dbRI + dbMI + kmTime + kmRI + kmMI
            # if retSum != 0.0:
            #     avgafTime = avgafTime + afTime
            #     avgafRI = avgafRI + afRI
            #     avgafMI = avgafMI + afMI
            #     avghirTime = avghirTime + hirTime
            #     avghirRI = avghirRI + hirRI
            #     avghirMI = avghirMI + hirMI
            #     avgdbTime = avgdbTime + dbTime
            #     avgdbRI = avgdbRI + dbRI
            #     avgdbMI = avgdbMI + dbMI
            #     avgkmTime = avgkmTime + kmTime
            #     avgkmRI = avgkmRI + kmRI
            #     avgkmMI = avgkmMI + kmMI
            #     testNo = testNo + 1
            
            # print 'doing the clustering'
            afTime, majHit, appHit, clusterArticles = myResultCluster.getClusters(queryString, trainData, testData, vectorizer, categories, queryResults, classCenter, nClusters, firstN)
            # print clusterArticles
            # if afTime != 0:
#                 testNo = testNo + 1
#                 majHitRate = majHitRate + majHit
#                 appHitRate = appHitRate + appHit
                # print
                # print
                # print queryString, 'was clustered'
                # print
                # print
            
        # print 'average afTime', avgafTime / testNo
        # print 'average afRI', avgafRI / testNo
        # print 'average afMI', avgafMI / testNo
        # print 'average hirTime', avghirTime / testNo
        # print 'average hirRI', avghirRI / testNo
        # print 'average hirMI', avghirMI / testNo
        # print 'average dbTime', avgdbTime / testNo
        # print 'average dbRI', avgdbRI / testNo
        # print 'average dbMI', avgdbMI / testNo
        # print 'average kmTime', avgkmTime / testNo
        # print 'average kmRI', avgkmRI / testNo
        # print 'average kmMI', avgkmMI / testNo
        
        # print 'number of hit is:', hitRate
        # print 'average hit rate is:', hitRate / testNo
        # return testNo
        # print 'average majority hit rate is:', majHitRate / testNo
        # print 'average appearance hit rate is:', appHitRate / testNo
        # print 'the number of majhit is:', majHitRate, ', and the number of test is:', testNo
            articleNo = 0
            for cluster, articleList in clusterArticles.iteritems():
                # print cluster
                if articleNo >= 10:
                    break
                for article in articleList:
                    print '###################################################################################################'
                    print article[0: len(article) / 5], '......(Read More)'
                    print
                    articleNo = articleNo + 1
                    if articleNo >= 5:
                        break
        return
    
    def getInput(self, prompt):
        while True:
            s = raw_input(prompt)
            return s