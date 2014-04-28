# classifierDescription, score, trainTime, testTime

# average afTime 5.27120323181
# average afRI 0.183442258531
# average afMI 0.305886318985
# average hirTime 12.4508558631
# average hirRI 0.164617401764
# average hirMI 0.288545043445
# average dbTime 12.7756240964
# average dbRI 8.51188446072e-15
# average dbMI 3.65691059595e-17
# average kmTime 6.98999812603
# average kmRI 0.164617401764
# average kmMI 0.288545043445
from plotRIMI import *

def main():
    results = []
    results.append(('Afinity Propagation', 5.27120323181, 0.183442258531, 0.305886318985))
    results.append(('Hierarchical Clustering', 12.4508558631, 0.164617401764, 0.288545043445))
    results.append(('K-Means Clustering', 6.98999812603, 0.164617401764, 0.288545043445))
    results.append(('DBSCAN', 12.7756240964, 8.51188446072e-15, 3.65691059595e-17))
    myPlot = plotRIMI()
    myPlot.plots(results)




if __name__ == "__main__":
    main()