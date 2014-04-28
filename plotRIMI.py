import pylab as pl
import numpy as np

class plotRIMI:
    def plots(self, results):
        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        (clf_names, timeSpan, RI, MI) = results
        RI = np.array(RI) / np.max(RI)
        MI = np.array(MI) / np.max(MI)
        timeSpan = np.array(timeSpan) / np.max(timeSpan)
        # pl.figure(figsize=(12, 8))
        pl.title('Clustering Algorithms Comparison')
        pl.barh(indices, timeSpan, .2, label='Time', color='r')
        pl.barh(indices + .25, RI, .2, label='Rand Index',
                color='g')
        pl.barh(indices + .5, MI, .2, label='Mutual Information', color='b')
        pl.yticks(())
        pl.legend(loc='best')
        pl.subplots_adjust(left=.25)
        pl.subplots_adjust(top=.95)
        pl.subplots_adjust(bottom=.05)

        for (i, c) in zip(indices, clf_names):
            pl.text(-.37, i, c)

        pl.show()