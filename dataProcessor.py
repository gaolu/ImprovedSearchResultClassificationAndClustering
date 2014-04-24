from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from time import time

class dataProcessor:
    def processData(self):
        return
    
    def trainFeatureExtract(self, opts, trainData, trainDataSize):
        print 'Extracting features from the training dataset using a sparse vectorizer'
        t0 = time()
        if opts.use_hashing:
            vectorizer = HashingVectorizer(stop_words='english', non_negative=True, n_features=opts.n_features)
            dataTrain = vectorizer.transform(trainData.data)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
            dataTrain = vectorizer.fit_transform(trainData.data)
        duration = time() - t0
        print 'done in %fs at %0.3fMB/s' % (duration, trainDataSize / duration)
        print 'n_samples: %d, n_features: %d' % dataTrain.shape
        print 
        return dataTrain, vectorizer
    
    def testFeatureExtract(self, opts, testData, testDataSize, vectorizer):
        print 'Extracting features from the test dataset using the same vectorizer'
        t0 = time()
        dataTest = vectorizer.transform(testData.data)
        duration = time() - t0
        print 'done in %fs at %0.3fMB/s' % (duration, testDataSize / duration)
        print 'n_samples: %d, n_features: %d' % dataTest.shape
        print
        return dataTest
    
    def chiFeatueSel(self, opts, dataTrain, labelTrain, dataTest):
        print('Extracting %d best features by a chi-squared test' % opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        dataTrain = ch2.fit_transform(dataTrain, labelTrain)
        dataTest = ch2.transform(dataTest)
        print('done in %fs' % (time() - t0))
        print
        return dataTrain, dataTest