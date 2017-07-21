from sklearn import datasets as skdatasets
from sklearn import metrics
from sklearn.datasets.mldata import fetch_mldata
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

digits_simple = skdatasets.load_digits()
digits_full = fetch_mldata("MNIST original")

# number_of_data = digits.data.shape[0]
# number_of_small_data = number_of_data // 2
# number_of_cross_validation_data = number_of_data // 4
# number_of_big_data = number_of_data * 3 // 4
# number_of_test_data = number_of_data // 4
#
# train_data_small = digits.data[0:number_of_small_data]
# cross_validation_data = digits.data[number_of_small_data:number_of_small_data+number_of_cross_validation_data]
# train_data_big = digits.data[0:number_of_big_data]
# test_data = digits.data[number_of_big_data:number_of_big_data+number_of_test_data]
# train_label_small = digits.target[0:number_of_small_data]
# cross_validation_label = digits.target[number_of_small_data:number_of_small_data+number_of_cross_validation_data]
# train_label_big = digits.target[0:number_of_big_data]
# test_label = digits.target[number_of_big_data:number_of_big_data+number_of_test_data]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
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
        train_test_split(X, y, test_size=.1, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("===================Now start %s for %s===================" % (name, dataset_names[ds_cnt]))
        print("fitting...")
        clf.fit(X_train, y_train)
        print("predicting...")
        predicted = clf.predict(X_test)
        print("evaluating...")
        print("%s accuracy: %s" % (name, metrics.accuracy_score(y_test, predicted)))
