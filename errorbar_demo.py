"""
Demo of the errorbar function.
"""
import numpy as np
import matplotlib.pyplot as plt

# example data
# x = np.arange(0.1, 4, 0.5)
# y = np.exp(-x)
# 
# plt.errorbar(x, y, xerr=0.2, yerr=0.4)
# plt.show()

# plt.figure(figsize=(12, 8))
plt.plot([0,1,2,3,4, 5, 6, 7, 8, 9, 10], [0,0.375714285714,0.414285714286,0.471428571429,0.471428571429, 0.471428571429, 0.471428571429, 0.471428571429, 0.471428571429, 0.471428571429, 0.471428571429], 'r--', lw=5, label='Cluster Center to Class Average')
plt.plot([0,1,2,3,4, 5, 6, 7, 8, 9, 10], [0,0.21714285714,0.284285714286,0.371428571429,0.371428571429, 0.371428571429, 0.371428571429, 0.371428571429, 0.371428571429, 0.371428571429, 0.371428571429], 'b--.', lw=5, label='Cluster Center to Class Center')
plt.plot([0,1,2,3,4, 5, 6, 7, 8, 9, 10], [0,0.371428571429,0.471428571429,0.671428571429,0.771428571429, 0.771428571429, 0.771428571429, 0.771428571429, 0.771428571429, 0.771428571429, 0.771428571429], 'g--', lw=5, label='Cluster Average to Class Average')
plt.plot([0,1,2,3,4, 5, 6, 7, 8, 9, 10], [0,0.485714285714,0.514285714286,0.571428571429,0.571428571429, 0.571428571429, 0.571428571429, 0.571428571429, 0.571428571429, 0.571428571429, 0.571428571429], 'k--.', lw=5, label='Cluster Average to Class Center')
plt.legend(loc='best')
plt.xlim([0, 10])
plt.title('Hit Rate Comparison')
plt.xlabel('Number of Clusters')
plt.ylabel('Hit Rate')
# plt.axis([0, 6, 0, 20])
plt.show()

