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

digits_simple = skdatasets.load_digits()
digits_full = fetch_mldata("MNIST original")
digits_full.data = digits_full.data[:10000]
digits_full.target = digits_full.target[:10000]
digits_full.cv_data = digits_full.data[10000:12000]
digits_full.cv_target = digits_full.data[10000:12000]

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
datasets = [(digits_simple.data, digits_simple.target), (digits_full.data, digits_full.target)]

# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("===================Now start %s for %s===================" % (name, dataset_names[ds_cnt]))
        output_name = "%s_%s.pkl" % (name, dataset_names[ds_cnt])
        if os.path.isfile(output_name):
            print("loading...")
            clf = joblib.load(output_name)[1]
        else:
            print("fitting...")
            clf.fit(X_train, y_train)
        print("predicting...")
        predicted = clf.predict(X_test)
        print("evaluating...")
        print("%s accuracy: %s" % (name, metrics.accuracy_score(y_test, predicted)))
        print("dumping weights...")
        joblib.dump((output_name, clf), output_name)
