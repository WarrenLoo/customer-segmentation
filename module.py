import scipy as ss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import seaborn as sns


def cramers_corrected_stat(confusion_matrix):
    """calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def plot_count_dist(con_col, cat_col, df):
    """_summary_

    Args:
        con_col (LIST): con_col consists of list of continous column names
        cat_col (LIST): cat_col consists of list of categorical column names
        df (_type_): _description_
    """
    # for continous
    for con in con_col:
        plt.figure()
        sns.distplot(df[con])
        plt.show()

    # for categorical
    for cat in cat_col:
        plt.figure()
        sns.countplot(x=df[cat])
        plt.show()


def predict_ml_models(X_train, X_test, y_train, y_test):
    """This method is to perform machine learning models prediction and training

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
    """
    # %% model development ML
    # ML Alogrithms:
    # KNN, DT, SVC, Random Forest, Logistic Regression

    # pipeline

    # KNN
    steps_mms_knn = Pipeline([("mms", MinMaxScaler()), ("knn", KNeighborsClassifier())])

    steps_ss_knn = Pipeline([("ss", StandardScaler()), ("knn", KNeighborsClassifier())])

    # DT
    steps_mms_dt = Pipeline([("mms", MinMaxScaler()), ("dt", DecisionTreeClassifier())])

    steps_ss_dt = Pipeline([("ss", StandardScaler()), ("dt", DecisionTreeClassifier())])

    # RF
    steps_mms_rf = Pipeline([("mms", MinMaxScaler()), ("rf", RandomForestClassifier())])

    steps_ss_rf = Pipeline([("ss", StandardScaler()), ("rf", RandomForestClassifier())])

    # LR
    steps_mms_lr = Pipeline([("mms", MinMaxScaler()), ("lr", LogisticRegression())])

    steps_ss_lr = Pipeline([("ss", StandardScaler()), ("lr", LogisticRegression())])

    # SVC
    steps_mms_svc = Pipeline([("mms", MinMaxScaler()), ("svc", SVC())])

    steps_ss_svc = Pipeline([("ss", StandardScaler()), ("svc", SVC())])

    pipelines = [
        steps_mms_dt,
        steps_ss_dt,
        steps_mms_knn,
        steps_ss_knn,
        steps_mms_lr,
        steps_ss_lr,
        steps_mms_rf,
        steps_ss_rf,
        steps_mms_svc,
        steps_ss_svc,
    ]

    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    best_accuracy = 0
    for i, model in enumerate(pipelines):
        if model.score(X_test, y_test) > best_accuracy:
            best_accuracy = model.score(X_test, y_test)
            best_pipeline = model

    print(
        "The best pipeline is {} with accuracy of {}".format(
            best_pipeline, best_accuracy
        )
    )

    # %% model analysis
    from sklearn.metrics import classification_report

    print(classification_report(y_test, model.predict(X_test)))
