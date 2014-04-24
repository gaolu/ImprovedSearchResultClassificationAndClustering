from sklearn.datasets import fetch_20newsgroups
class dataLoader:
    def loadData(self, opts):
        if opts.all_categories:
            categories = None
        else:
            categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics',
                          'sci.space']

        if opts.filtered:
            remove = ('headers', 'footers', 'quotes')
        else:
            remove = ()

        print('Loading 20 newsgroups dataset for categories:')
        print((categories if categories else 'all'))

        data_train = fetch_20newsgroups(subset='train', categories=categories,
                                        shuffle=True, random_state=42,
                                        remove=remove)

        data_test = fetch_20newsgroups(subset='test', categories=categories,
                                       shuffle=True, random_state=42,
                                       remove=remove)
        
        categories = data_train.target_names  # for case categories == None
        # print(len(data_train))
        print('data loaded')
        
        return data_train, data_test, categories
    
    def sizeMb(self, documents):
        return sum(len(s.encode('utf-8')) for s in documents) / 1e6
        
    def printFileSize(self, trainData, trainDataSize, testData, testDataSize, categories):
        print('%d documents - %0.3fMB (training set)' % (len(trainData.data), trainDataSize))
        print('%d documents - %0.3fMB (test set)' % (len(testData.data), testDataSize))
        print('%d categories' % len(categories))
        print