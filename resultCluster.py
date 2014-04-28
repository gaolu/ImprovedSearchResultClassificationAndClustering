import numpy as np
import pylab as pl
import itertools
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Ward
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics
from time import time
from scipy.spatial.distance import sqeuclidean
from collections import OrderedDict
from numpy import array
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
class resultCluster:
    def getQueryResult(self, queryString, trainData, testData):
        # print 'the query string is:', queryString.lower()
        wordList = [word for word in queryString.lower().split(" ") if not word in stopwords.words('english')]
        # print len(wordList)
        trainArticleNoList = []
        for word in wordList:
            # print 'current word is: ', word
            trainArticleNoList.extend(self.findElementInList(word, trainData.data))
            # print len(trainArticleNoList)
        # print len(trainArticleNoList)
        trainArticleList = list(set(trainArticleNoList))
        trainArticleList.sort()
        # print 'after removing duplicates', len(trainArticleList)
        trainArticleDataList = [trainData.data[i] for i in trainArticleList]
        trainArticleTargetList = [trainData.target[i] for i in trainArticleList]
        # print 'the label list has length:', len(trainArticleTargetList), len(trainArticleDataList)
        # print trainArticleNoList
        return trainArticleDataList, trainArticleTargetList
    
    def findElementInList(self, element, srcList):
        # return a list of indexes
        indexes = [item for item in range(len(srcList)) if element in srcList[item]]
        return indexes
    
    def mostCommon(self, srcList):
        return max(set(srcList), key=srcList.count)
    
    def getDistance(self, mat1, mat2):
        # mat1 and mat2 are csr matrix
        #print len(mat1), len(mat2.todense())
        return np.linalg.norm(array(mat1) - mat2.toarray()[0])
    
    def getClusterClassDistance(self, queryResults, trainData, vectorizer, categories, clusters, srcDataList, srcTargetList, classCenter):
        # queryResults is a number
        distanceDict = {}
        srcDataListVec = vectorizer.transform(srcDataList)
        #print srcDataListVec.shape
        # clusterVec = {}
        # calculate distances and average them
        for category, fileList in clusters.iteritems():
            distance = 0.0
            for file in fileList:
                distance = distance + self.getDistance(classCenter[queryResults], srcDataListVec[file])
            distance = distance / len(fileList)
            distanceDict[category] = distance
        return distanceDict
    
    
    def getClusters(self, queryString, trainData, testData, vectorizer, categories, queryResults, classCenter, nClusters, firstN):
        srcDataList, srcTargetList = self.getQueryResult(queryString, trainData, testData)
        
        # we don't need to predefine the nClusters
        # but just to avoid clustering if the number of articles retrieved is smaller than a fixed number
        # if len(srcDataList) < nClusters:
        #     # return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        #     # print 'the number of articles retrieved is:', len(srcDataList)
        #     return 0.0, 0.0, 0.0
        if len(srcDataList) == 0:
            return 0.0, 0.0, 0.0, {}
        # print 'the length of the retrieved list is:', len(srcDataList)
        vecDataList = vectorizer.transform(srcDataList)
        # print type(vecDataList), vecDataList.shape
        # print 'the length of the data vector is:', len(vecDataList.toarray())
        # print 'the length of each of the data vector is:', len(vecDataList.toarray()[0])
        
        # print
        # print
        # print
        
        
        # excercise propagation affinity algorithm
        print '===============doing the affinity propogation=============='
        # print
        # print len(vecDataList.toarray()), len(vecDataList.toarray()[1])
        # print
        st = time()
        af = AffinityPropagation().fit(vecDataList.toarray())
        # print af
        cluster_centers_indices = af.cluster_centers_indices_
        # print 'cluster centers indices is:'
        # print cluster_centers_indices
        labels = af.labels_
        # print 'labels are:'
        # print labels
        n_clusters_ = len(cluster_centers_indices)
        print 'Elapsed time:', time() - st
        afTime = time() - st
        print('Estimated number of clusters: %d' % n_clusters_)
        # print("Adjusted Rand Index: %0.3f"
        #       % metrics.adjusted_rand_score(srcTargetList, labels))
        afRI = metrics.adjusted_rand_score(srcTargetList, labels)
        # print("Adjusted Mutual Information: %0.3f"
        #       % metrics.adjusted_mutual_info_score(srcTargetList, labels))
        afMI = metrics.adjusted_mutual_info_score(srcTargetList, labels)
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(vecDataList.toarray(), labels, metric='sqeuclidean'))
        # print
        # print
        # print
        
        # print '=================doing post cluster processing to find the best cluster(s)================='
        
        # print '+++++++++++++putting articles to clusters+++++++++++++'
        clusters = {}
        for i in range(len(labels)):
            if labels[i] in clusters:
                clusters[labels[i]].append(i)
            else:
                articleList = []
                articleList.append(i)
                clusters[labels[i]] = articleList
        # print len(clusters)
        # print clusters
        
        
        # print
        # print '++++++++++++getting original article labels++++++++++++'
        clusterLabels = {}
        for i in range(len(labels)):
            if labels[i] in clusterLabels:
                clusterLabels[labels[i]].append(srcTargetList[i])
            else:
                articleList = []
                articleList.append(srcTargetList[i])
                clusterLabels[labels[i]] = articleList
        # print clusterLabels
        
        
        
        # print
        # print '+++++++++++++calculating distances from each cluster to the predicted class++++++++++++++++'
        distanceDict = self.getClusterClassDistance(queryResults, trainData, vectorizer, categories, clusters, srcDataList, srcTargetList, classCenter)
        orderedDistanceDict = OrderedDict(sorted(distanceDict.items(), key=lambda t: t[1]))
        # print orderedDistanceDict
        
        
        
        # print
        # print '+++++++++++calculating distance from each cluster center point to the predicted class++++++++++'
        clusterCenters = {}
        for clusterNo in range(len(cluster_centers_indices)):
            clusterCenters[clusterNo] = [cluster_centers_indices[clusterNo]]
        # print clusterCenters
        centerDistanceDict = self.getClusterClassDistance(queryResults, trainData, vectorizer, categories, clusterCenters, srcDataList, srcTargetList, classCenter)
        orderedCenterDistanceDict = OrderedDict(sorted(centerDistanceDict.items(), key=lambda t: t[1]))
        # print orderedCenterDistanceDict
        
        
        
        # print
        # print '++++++++++calculating majority vote on each cluster+++++++++++++'
        majVoteDict = {}
        for cluster, labelList in clusterLabels.iteritems():
            maj = self.mostCommon(labelList)
            majVoteDict[cluster] = maj
        # print majVoteDict
        
        
        # print
        # print '++++getting top n clusters++++'
        if n_clusters_ < firstN:
            topN = itertools.islice(orderedCenterDistanceDict.items(), 0, len(orderedCenterDistanceDict))
        else:
            topN = itertools.islice(orderedCenterDistanceDict.items(), 0, firstN)
        
        
        # print 'topN clusters are:'
        # print topN
        
        
        # print
        # print '+++++++++calculating hit rate with majority vote+++++++++++++'
        majHit = 0
        
        # for cluster, value in topN:
        #     # print 'in topN, cluster number is:', cluster
        #     # print 'in topN, value is:', value
        #     if majVoteDict[cluster] == queryResults:
        #         majHit = 1
        #         # print 'maj hit!!!!!!!!!!!!!!!!!!'
        #         break
        
        
        # print '+++++++++++calculating hit rate with indicator of appearance++++++++++'
        appHit = 0
        
        # for cluster, value in topN:
        #     if queryResults in clusterLabels[cluster]:
        #         appHit = 1
        #         print 'app hit!!!!!!!!!!!!!!!!!!'
        #         break
        
        
        # print
        # print
        # print
        
        
        print
        print '+++++++++returnning the topN clusters of articles'
        clusterArticles = {}
        for cluster, value in topN:
            articleList = [srcDataList[articleNo] for articleNo in clusters[cluster]]
            clusterArticles[cluster] = articleList
        return afTime, majHit, appHit, clusterArticles
        
        
        
        # print '=================doing hierarchical clustering===================='
#         # Compute clustering
#         print("Compute unstructured hierarchical clustering...")
#         st = time()
#         ward = Ward(n_clusters=nClusters).fit(vecDataList.toarray())
#         label = ward.labels_
#         # print("Elapsed time: ", time() - st)
#         hirTime = time() -st
#         # print("Number of points: ", label.size)
#         # print("Adjusted Rand Index: %0.3f"
#         #       % metrics.adjusted_rand_score(srcTargetList, label))
#         hirRI = metrics.adjusted_rand_score(srcTargetList, label)
#         # print("Adjusted Mutual Information: %0.3f"
#         #       % metrics.adjusted_mutual_info_score(srcTargetList, label))
#         hirMI = metrics.adjusted_mutual_info_score(srcTargetList, label)
#         print
#         print
#         print
#         
#         print '====================doing DBSCAN========================'
#         st = time()
#         # X = StandardScaler().fit_transform(vecDataList.toarray())
#         X = vecDataList.toarray()
#         # Compute DBSCAN
#         db = DBSCAN(eps=0.3, min_samples=10).fit(X)
#         core_samples = db.core_sample_indices_
#         labels = db.labels_
# 
#         # Number of clusters in labels, ignoring noise if present.
#         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#         # print("Elapsed time: ", time() - st)
#         dbTime = time() - st
#         # print('Estimated number of clusters: %d' % n_clusters_)
#         # print("Homogeneity: %0.3f" % metrics.homogeneity_score(srcTargetList, labels))
#         # print("Completeness: %0.3f" % metrics.completeness_score(srcTargetList, labels))
#         # print("V-measure: %0.3f" % metrics.v_measure_score(srcTargetList, labels))
#         # print("Adjusted Rand Index: %0.3f"
#         #       % metrics.adjusted_rand_score(srcTargetList, labels))
#         dbRI = metrics.adjusted_rand_score(srcTargetList, labels)
#         # print("Adjusted Mutual Information: %0.3f"
#         #       % metrics.adjusted_mutual_info_score(srcTargetList, labels))
#         dbMI = metrics.adjusted_mutual_info_score(srcTargetList, labels)
#         
#         
#         
#         # print '=================doing mean shift========================='
#         # # Compute clustering with MeanShift
#         # 
#         # # The following bandwidth can be automatically detected using
#         # bandwidth = estimate_bandwidth(vecDataList.toarray(), quantile=0.2, n_samples=500)
#         # 
#         # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#         # ms.fit(vecDataList.toarray())
#         # labels = ms.labels_
#         # cluster_centers = ms.cluster_centers_
#         # 
#         # labels_unique = np.unique(labels)
#         # n_clusters_ = len(labels_unique)
#         # 
#         # print("number of estimated clusters : %d" % n_clusters_)
#         # print("Adjusted Rand Index: %0.3f"
#         #       % metrics.adjusted_rand_score(srcTargetList, labels))
#         # print("Adjusted Mutual Information: %0.3f"
#         #       % metrics.adjusted_mutual_info_score(srcTargetList, labels))
#         # print
#         # print
#         # print
#         
#         
#         
#         
#         
#         print '===============doing the kMeans========================='
#         # kMeans algorithm
#         # estimators = {'KMeans 10': KMeans(n_clusters=nClusters)}
#         
#         est = KMeans(n_clusters=nClusters)
#         # print description
#         # print 'using the training data'
#         # time0 = time()
#         # est.fit(vectorizer.transform(trainData.data))
#         # labels = est.labels_
#         # timeSpan = time() - time0
#         # print 'RI is:', metrics.adjusted_rand_score(trainData.target, labels)
#         # print 'time is:', timeSpan, 'seconds'
#         
#         
#         
#         time0 = time()
#         est.fit(vecDataList)
#         labels = est.labels_
#         timeSpan = time() - time0
#         print 'Elapsed time:', timeSpan
#         kmTime = timeSpan
#         # print("Adjusted Rand Index: %0.3f"
#         #       % metrics.adjusted_rand_score(srcTargetList, label))
#         kmRI = metrics.adjusted_rand_score(srcTargetList, label)
#         # print("Adjusted Mutual Information: %0.3f"
#         #       % metrics.adjusted_mutual_info_score(srcTargetList, label))
#         kmMI = metrics.adjusted_mutual_info_score(srcTargetList, label)
        # clusterLabels = labels.astype(np.float)
#         clusters = {}
#         for i in range(len(clusterLabels)):
#             if clusterLabels[i] in clusters:
#                 clusters[clusterLabels[i]].append(i)
#             else:
#                 clusterList = []
#                 clusterList.append(i)
#                 clusters[clusterLabels[i]] = clusterList
#         # print clusters
#         
#         # calculate the distance between clusters and the predicted query class
#         clusterDistanceDict = self.getClusterClassDistance(queryResults, trainData, vectorizer, categories, clusters, srcDataList, srcTargetList, classCenter)
#         # print clusterDistanceDict
#         orderedClusterDistanceDict = OrderedDict(sorted(clusterDistanceDict.items(), key=lambda t: t[1], reverse=True))
#         # print orderedClusterDistanceDict
#         firstN = 5
#         topN = itertools.islice(orderedClusterDistanceDict.items(), 0, firstN)
#         hit = 0
#         for key, value in topN:
#             if key == queryResults:
#                 # print 'True'
#                 hit = 1
#                 break
            # clusterClasses = {}
            # for key, val in clusters.iteritems():
            #     # classes = self.mostCommon([srcTargetList[i] for i in val])
            #     print [srcTargetList[i] for i in val]
            #     classes = self.mostCommon([categories[srcTargetList[i]] for i in val])
            #     clusterClasses[key] = classes
            # print clusterClasses
        # return afTime, afRI, afMI, hirTime, hirRI, hirMI, dbTime, dbRI, dbMI, kmTime, kmRI, kmMI
        # return afTime, afRI, afMI