import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile('weight.npz')
number_of_algos = 8
data = data.reshape([102, number_of_algos])
data = data[2:][:]
data = np.insert(data, 0, values=[1/number_of_algos]*number_of_algos, axis=0)
X = np.linspace(0, 10000, 101)

names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


for i in range(0, number_of_algos):
    print(i)
    y = [d[i] for d in data]
    plt.plot(X, y, label=names[i])

plt.legend()
plt.savefig("weights.png", dpi=72)
plt.show()
