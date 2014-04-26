from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import metrics
from sklearn.utils.extmath import density
from time import time
import numpy as np

class classifiers:
    def getResults(self, opts, dataTrain, labelTrain, dataTest, labelTest, featureNames, categories):
        results = []
        self.basicClassifiers(opts, results, dataTrain, labelTrain, dataTest, labelTest, featureNames, categories)
        return results
        
    def basicClassifiers(self, opts, results, dataTrain, labelTrain, dataTest, labelTest, featureNames, categories):
        for (classifier, name) in ((RidgeClassifier(tol=0.01, solver='lsqr'), 'Ridge Classifier'), 
                            (Perceptron(n_iter=50), 'Perceptron'),
                            (PassiveAggressiveClassifier(n_iter=50), 'Passive-Aggressive'),
                            (KNeighborsClassifier(n_neighbors=10), 'kNN')):
            print('=' * 80)
            print(name)
            results.append(self.benchmark(opts, classifier, dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))
            
        for penalty in ['l2', 'l1']:
            print('=' * 80)
            print('%s penalty' % penalty.upper())

            # Train Liblinear model

            results.append(self.benchmark(opts, LinearSVC(loss='l2', penalty=penalty,
                           dual=False, tol=1e-3), dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))

            # Train SGD model

            results.append(self.benchmark(opts, SGDClassifier(alpha=.0001, n_iter=50,
                           penalty=penalty), dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))

        # Train SGD with Elastic Net penalty

        print('=' * 80)
        print('Elastic-Net penalty')
        results.append(self.benchmark(opts, SGDClassifier(alpha=.0001, n_iter=50,
                       penalty='elasticnet'), dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))

        # Train NearestCentroid without threshold

        print('=' * 80)
        print('NearestCentroid (aka Rocchio classifier)')
        results.append(self.benchmark(opts, NearestCentroid(), dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))

        # Train sparse Naive Bayes classifiers

        print('=' * 80)
        print('Naive Bayes')
        results.append(self.benchmark(opts, MultinomialNB(alpha=0.01), dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))
        results.append(self.benchmark(opts, BernoulliNB(alpha=0.01), dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))
        
        print('=' * 80)
        print('LinearSVC with L1-based feature selection')
        results.append(self.benchmark(opts, L1LinearSVC(), dataTrain, labelTrain, dataTest, labelTest, featureNames, categories))
        return
        
    def benchmark(self, opts, classifier, dataTrain, labelTrain, dataTest, labelTest, featureNames, categories):
        print '_' * 80
        print 'Training: '
        print classifier
        
        # start training and measure the time frame
        t0 = time()
        classifier.fit(dataTrain, labelTrain)
        trainTime = time() - t0
        print 'train time: %0.3fs' % trainTime
        
        # start prediction and measure the time frame
        t0 = time()
        predictor = classifier.predict(dataTest)
        testTime = time() - t0
        print 'test time: %0.3fs' % testTime
        
        # accuracy
        score = metrics.f1_score(labelTest, predictor)
        print 'f1 score: %0.3f' % score
        
        if hasattr(classifier, 'coef_'):
            print 'dimensionality: %d' % classifier.coef_.shape[1]
            print 'density: %f' % density(classifier.coef_)
            
            if opts.print_top10 and featureNames is not None:
                print 'top 10 keywords per class:'
                for i, category in enumerate(categories):
                    top10 = np.argsort(classifier.coef_[i])[-10 : ]
                    print self.trim('%s: %s' % (category, ' '.join(featureNames[top10])))
            print 
        
        if opts.print_report:
            print 'classification report:'
            print metrics.classification_report(labelTest, predictor, target_names = categories)
        
        if opts.print_cm:
            print 'confusion matrix:'
            print metrics.confusion_matrix(labelTest, predictor)
        
        print
        classifierDescription = str(classifier).split('(')[0]
        return classifierDescription, score, trainTime, testTime
        
    def trim(self, s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return (s if len(s) <= 80 else s[:77] + '...')
    
    def queryClassifier(self, opts, classifier, dataTrain, labelTrain, dataTest, featureNames, categories):
        
        # start training and measure the time frame
        t0 = time()
        classifier.fit(dataTrain, labelTrain)
        trainTime = time() - t0
        print 'train time: %0.3fs' % trainTime
        
        # start prediction and measure the time frame
        t0 = time()
        predictor = classifier.predict(dataTest)
        # print type(predictor), len(predictor), predictor
        testTime = time() - t0
        print 'test time: %0.3fs' % testTime
        print 'Predicted category: ' + categories[predictor[0]], 'the number of category is:', predictor[0]
        return predictor[0]
        
        

class L1LinearSVC(LinearSVC):

    def fit(self, X, y):

        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.

        self.transformer_ = LinearSVC(penalty='l1', dual=False,
                tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)