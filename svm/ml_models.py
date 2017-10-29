from __future__ import print_function
import logging
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import data_processing.data_processing as dp

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# #############################################################################
# Load some categories from the training se
# writes coef_d feat_d coef_r feat_r section model acc_d acc_r acc_all recall f1 support

with open("svm_top_features.tsv", "w", encoding="utf-8") as tsv:
    tsv.write("coef_d\t" +
              "feat_d\t" +
              "coef_r\t" +
              "feat_r\t" +
              "section\t" +
              "model\t" +
              "acc_d\t" +
              "acc_r\t" +
              "acc_all\t" +
              "recall\t" +
              "f1\t" +
              "support"
              )
    tsv.close()

sections = ["2015_1q", "2015_2q", "2015_3q", "2015_4q", "2016_1q", "2016_2q", "2016_3q", "2016_4q", "2017_1q", "2017_2q", "2017_2q_REDUCED", "2017_2q_REDUCED"]

data = dp.Data_Processing(load_tokenizers=True)
for i, section in enumerate(sections):
    print("\nWORKING ON SECTION: ", section, "\n")
    data_train, y_train, data_test, y_test, _, _ = data.run(
        train_file="../data/train_data/train_data.p",
        test_file="../data/test_data/test_data.p", section=i + 1)

    target_names = list(set(y_train))

    print("Extracting features from the training data using a sparse vectorizer")

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train)

    print("Extracting features from the test data using the same vectorizer")
    X_test = vectorizer.transform(data_test)

    feature_names = vectorizer.get_feature_names()


    # #############################################################################
    # Benchmark classifiers
    def benchmark(clf, section):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.4f" % score)

        print("classification report:")
        class_matrix = metrics.classification_report(y_test, pred,target_names=target_names).split()

        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

        print()

        clf_descr = str(clf).split('(')[0]

        if hasattr(clf, 'coef_'):
            with open("svm_top_features.tsv", "a", encoding="utf-8") as tsv2:

                print("dimensionality: %d" % clf.coef_.shape[1])
                print("density: %f" % density(clf.coef_))

                coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
                top = zip(coefs_with_fns[:50], coefs_with_fns[:-(50 + 1):-1])
                for (coef_1, fn_1), (coef_2, fn_2) in top:
                    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))
                    # writes coef_d feat_d coef_r feat_r section model acc_d acc_r acc_all recall f1 support
                    tsv2.write("\n%.5f\t%s\t%.5f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" %
                              (coef_1, fn_1, coef_2, fn_2, section, str(clf).split("(")[0],
                               class_matrix[5], class_matrix[10], class_matrix[17],
                               class_matrix[18], class_matrix[19], class_matrix[20]))
                tsv2.close()

            print()

        return clf_descr, score, train_time, test_time


    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, section))

    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                           tol=1e-3), section))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty), section))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet"), section))

    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid(), section))

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01), section))
    results.append(benchmark(BernoulliNB(alpha=.01), section))

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                        tol=1e-3))),
        ('classification', LinearSVC(penalty="l2"))]), section))
