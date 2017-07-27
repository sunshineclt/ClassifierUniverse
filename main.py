import math
import numpy as np
import os
from sklearn import datasets as skdatasets
from sklearn import metrics
from sklearn.datasets.mldata import fetch_mldata
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# load data and segment some cross validation part
digits_simple = skdatasets.load_digits()
digits_simple.cv_data = digits_simple.data[1400:]
digits_simple.cv_target = digits_simple.target[1400:]
digits_simple.data = digits_simple.data[:1400]
digits_simple.target = digits_simple.target[:1400]
digits_full = fetch_mldata("MNIST original")
digits_full.cv_data = digits_full.data[20000:22000]
digits_full.cv_target = digits_full.target[20000:22000]
digits_full.data = digits_full.data[:20000]
digits_full.target = digits_full.target[:20000]


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

dataset_names = ["digits_simple", "digits_full"]
datasets = [digits_simple, digits_full]

# iterate over datasets
for ds_cnt, data in enumerate(datasets):
    # cross validation part
    # preprocess dataset, split into training and test part
    X = data.data
    y = data.target
    X_cv = data.cv_data
    y_cv = data.cv_target
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

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
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)
    X_train = np.append(X_train, data.cv_data, axis=0)
    y_train = np.append(y_train, data.cv_target, axis=0)
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
    for x, y in zip(X_test, y_test):
        p_EWA = 0
        sum_weight = 0
        z = 0
        for algo_cnt in range(0, len(classifiers)):
            f[algo_cnt] = classifiers[algo_cnt].predict(x.reshape(1, x.shape[0]))[0]
            p_EWA += weight[algo_cnt] * f[algo_cnt]
            sum_weight += weight[algo_cnt]
            z += weight[algo_cnt]

        p_EWA /= sum_weight
        p_EWA = round(p_EWA)
        if p_EWA == y:
            accurate_count_EWA += 1
        p_REWA = f[np.random.choice(np.arange(0, len(classifiers)), p=[w/z for w in weight])]
        if p_REWA == y:
            accurate_count_REWA += 1
        for algo_cnt in range(0, len(classifiers)):
            if f[algo_cnt] != y:
                weight[algo_cnt] *= math.exp(-eta)

    accurate_count_EWA /= len(y_test)
    accurate_count_REWA /= len(y_test)
    print("test accuracy EWA: %f" % accurate_count_EWA)
    print("test accuracy REWA: %f" % accurate_count_REWA)
