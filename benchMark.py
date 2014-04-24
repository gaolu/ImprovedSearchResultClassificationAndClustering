from time import time
from sklearn import metrics
from sklearn.utils.extmath import density
class benchMark:
    def benchmark(self, opts, classifier, dataTrain, labelTrain, dataTest, labelTest):
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
            
            if opts.print_top10 and feature_name is not None:
                print 'top 10 keywords per class:'
                for i, category in enumerate(categories):
                    top10 = np.argsort(classifier.coef_[i])[-10 : ]
                    print trim('%s: %s' % (category, ' '.join(featureNames[top10])))
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