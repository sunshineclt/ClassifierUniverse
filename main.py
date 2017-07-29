import math
import os
import time, datetime

import numpy as np
from sklearn import datasets as skdatasets
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# load data and segment some cross validation part
digits_simple = skdatasets.load_digits()
digits_simple.cv_data = digits_simple.data[1400:]
digits_simple.cv_target = digits_simple.target[1400:]
digits_simple.test_data = digits_simple.data[1200:1400]
digits_simple.test_target = digits_simple.target[1200:1400]
digits_simple.train_data = digits_simple.data[:1200]
digits_simple.train_target = digits_simple.target[:1200]
# digits_full = fetch_mldata("MNIST original")
# digits_full.cv_data = digits_full.data[30000:35000]
# digits_full.cv_target = digits_full.target[30000:35000]
# digits_full.data = digits_full.data[:30000]
# digits_full.target = digits_full.target[:30000]
digits_full = np.load('mnist.npz')
digits_full.data = np.concatenate([digits_full['x_train'], digits_full['x_valid']], axis=0)
digits_full.target = np.concatenate([digits_full['y_train'], digits_full['y_valid']]).astype(np.int32)
digits_full.test_data = digits_full['x_test']
digits_full.test_target = digits_full['y_test'].astype(np.int32)

# select labeled data
rng = np.random.RandomState(1234)
data_rng = np.random.RandomState(4321)
inds = data_rng.permutation(digits_full.data.shape[0])
digits_full.data = digits_full.data[inds]
digits_full.target = digits_full.target[inds]
trainx = []
trainy = []
cvx = []
cvy = []
for j in range(10):
    trainx.append(digits_full.data[digits_full.target == j][:500*(j+1)])
    trainy.append(digits_full.target[digits_full.target == j][:500*(j+1)])
    cvx.append(digits_full.data[digits_full.target == j][5000:])
    cvy.append(digits_full.target[digits_full.target == j][5000:])
digits_full.train_data = np.concatenate(trainx, axis=0)
digits_full.train_target = np.concatenate(trainy, axis=0)
digits_full.cv_data = np.concatenate(cvx, axis=0)
digits_full.cv_target = np.concatenate(cvy, axis=0)

names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

dataset_names = ["digits_simple", "digits_full"]
datasets = [digits_simple, digits_full]

print("Start time: %s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

# iterate over datasets
for ds_cnt, data in enumerate(datasets):
    # cross validation part
    # preprocess dataset, split into training and test part
    X_train = data.train_data
    y_train = data.train_target
    X_cv = data.cv_data
    y_cv = data.cv_target
    X_test = data.test_data
    y_test = data.test_target

    cross_validation_max = 0
    chosen_algo_cnt = 0
    chosen_algo = classifiers[0]
    # iterate over classifiers
    for algo_cnt, algo in enumerate(zip(names, classifiers)):
        name, clf = algo
        print("===================Now start %s for %s cv===================" % (name, dataset_names[ds_cnt]))
        output_name = "%s_%s_cv.pkl" % (name, dataset_names[ds_cnt])
        if os.path.isfile(output_name):
            print("loading...")
            clf = joblib.load(output_name)[1]
        else:
            print("fitting...")
            clf.fit(X_train, y_train)
            print("dumping weights...")
            joblib.dump((output_name, clf), output_name)
        print("predicting...")
        predicted = clf.predict(X_cv)
        print("evaluating...")
        accuracy = float(metrics.accuracy_score(y_cv, predicted))
        print("%s cross validation accuracy: %f" % (name, accuracy))
        if accuracy > cross_validation_max:
            cross_validation_max = accuracy
            chosen_algo_cnt = algo_cnt
            chosen_algo = clf

    print("*****************CV chosen algo is %s*****************" % (names[chosen_algo_cnt]))
    print("predicting...")
    predicted = chosen_algo.predict(X_test)
    print("evaluating...")
    accuracy_cv = float(metrics.accuracy_score(y_test, predicted))
    print("test accuracy cv: %f" % accuracy_cv)

    # EWA part
    X_train = np.append(X_train, X_cv, axis=0)
    y_train = np.append(y_train, y_cv, axis=0)
    for algo_cnt in range(0, len(classifiers)):
        print("===================Now start %s for %s EWA===================" % (names[algo_cnt], dataset_names[ds_cnt]))
        output_name = "%s_%s_EWA.pkl" % (names[algo_cnt], dataset_names[ds_cnt])
        if os.path.isfile(output_name):
            print("loading...")
            classifiers[algo_cnt] = joblib.load(output_name)[1]
        else:
            print("fitting...")
            classifiers[algo_cnt].fit(X_train, y_train)
            print("dumping weights...")
            joblib.dump((output_name, classifiers[algo_cnt]), output_name)

    weight = [1] * len(classifiers)
    f = [0] * len(classifiers)
    accurate_count_EWA = 0
    accurate_count_REWA = 0
    eta = math.sqrt(8 * np.log(len(classifiers)) / len(y_test))
    print("*****************Start EWA gaming*****************")
    t = 0
    for x, y in zip(X_test, y_test):
        p_EWA = 0
        sum_weight = 0
        z = 0
        for algo_cnt in range(0, len(classifiers)):
            f[algo_cnt] = classifiers[algo_cnt].predict(x.reshape(1, x.shape[0]))[0]
            p_EWA += weight[algo_cnt] * f[algo_cnt]
            sum_weight += weight[algo_cnt]
            z += weight[algo_cnt]

        t += 1
        if t % 1000 == 0:
            print(weight)
        p_EWA /= sum_weight
        p_EWA = round(p_EWA)
        if p_EWA == y:
            accurate_count_EWA += 1
        p_REWA = f[np.random.choice(np.arange(0, len(classifiers)), p=[w/z for w in weight])]
        if p_REWA == y:
            accurate_count_REWA += 1
        z = 0
        for algo_cnt in range(0, len(classifiers)):
            if f[algo_cnt] != y:
                weight[algo_cnt] *= math.exp(-eta)
            z += weight[algo_cnt]
        weight = [x/z for x in weight]

    accurate_count_EWA /= len(y_test)
    accurate_count_REWA /= len(y_test)
    print("test accuracy EWA: %f" % accurate_count_EWA)
    print("test accuracy REWA: %f" % accurate_count_REWA)

print("End time: %s" % datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
