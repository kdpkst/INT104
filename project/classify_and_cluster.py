import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.metrics import classification_report, silhouette_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,normalized_mutual_info_score
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN


def data_pre_processing(dataframe):
    # preprocessing: remove ID columns, null rows and programme 0
    dataframe.drop(labels="ID", axis=1, inplace=True)
    dataframe.drop(dataframe[dataframe["Programme"].isnull()].index, inplace=True)
    dataframe.drop(dataframe[dataframe["Programme"] == 0].index, axis=0, inplace=True)


def print_describe(dataframe):
    # print mean,std,min,max and so on
    print(dataframe.describe())


def correlation(dataframe):
    correlations = dataframe.corr(method='pearson')
    print(correlations)


def data_distribution(dataframe):
    count_prog = dataframe.groupby("Programme").size()
    print(count_prog)


def data_skew(dataframe):
    skew = dataframe.skew()
    print(skew)


def histogram1(dataframe):
    dataframe.hist()
    plt.show()


def histogram2(dataframe):
    # count the number of students in different programme
    count = dataframe['Programme'].value_counts()
    plt.figure(figsize=(4.5, 4.5))
    plt.title("number of students in different programme")
    plt.xlabel("programme")
    for i in range(1, 5):
        plt.bar(str(i), count[i])
    plt.show()


def density_plot(dataframe):
    dataframe.plot(kind="density", subplots="True", layout=(3,2), sharex=False)
    plt.show()


def box_whisker(dataframe):
    dataframe.plot(kind="box", subplots="True", layout=(3, 2), sharex=False, sharey=False)
    plt.show()


def correlation_matrix_plot(dataframe):
    correlations = dataframe.corr()
    name = ['Q1', 'Q2', 'Q3', 'Q4', 'prog']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 5, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(name)
    ax.set_yticklabels(name)
    plt.show()


def scatter_matrix_plot(dataframe):
    pd.plotting.scatter_matrix(dataframe)
    plt.show()


def programme_classification(list1, list2, list3, list4, list5, dataframe):
    scoreMatrix = dataframe.values
    for i in range(len(scoreMatrix)):
        if scoreMatrix[i,5] == 1:
            list1.append(scoreMatrix[i, :5])
        elif scoreMatrix[i,5] == 2:
            list2.append(scoreMatrix[i, :5])
        elif scoreMatrix[i,5] == 3:
            list3.append(scoreMatrix[i, :5])
        elif scoreMatrix[i,5] == 4:
            list4.append(scoreMatrix[i, :5])
    for i in range(len(scoreMatrix)):
        list5.append(scoreMatrix[i,5])


def scatter_median(list1, list2, list3, list4):
    plt.figure(figsize=(4.5, 4.5))
    plt.title("median value")
    plt.scatter(np.zeros(len(list1)) + 1, np.median(np.array(list1), -1))
    plt.scatter(np.zeros(len(list2)) + 2, np.median(np.array(list2), -1))
    plt.scatter(np.zeros(len(list3)) + 3, np.median(np.array(list3), -1))
    plt.scatter(np.zeros(len(list4)) + 4, np.median(np.array(list4), -1))
    plt.xlabel("programme")
    plt.show()


def scatter_mean(list1, list2, list3, list4):
    plt.figure(figsize=(4.5, 4.5))
    plt.title("mean value")
    plt.scatter(np.zeros(len(list1)) + 1, np.mean(np.array(list1), -1))
    plt.scatter(np.zeros(len(list2)) + 2, np.mean(np.array(list2), -1))
    plt.scatter(np.zeros(len(list3)) + 3, np.mean(np.array(list3), -1))
    plt.scatter(np.zeros(len(list4)) + 4, np.mean(np.array(list4), -1))
    plt.xlabel("programme")
    plt.show()


def scatter_std(list1, list2, list3, list4):
    # draw the scatter of standard deviation of each student
    plt.figure(figsize=(4.5, 4.5))
    plt.title("standard deviation")
    plt.scatter(np.zeros(len(list1)) + 1, np.std(np.array(list1), -1))
    plt.scatter(np.zeros(len(list2)) + 2, np.std(np.array(list2), -1))
    plt.scatter(np.zeros(len(list3)) + 3, np.std(np.array(list3), -1))
    plt.scatter(np.zeros(len(list4)) + 4, np.std(np.array(list4), -1))
    plt.xlabel("programme")
    plt.show()


def scatter_min(list1, list2, list3, list4):
    plt.figure(figsize=(4.5, 4.5))
    plt.title("min score of each student")
    plt.scatter(np.zeros(len(list1)) + 1, np.min(np.array(list1), -1))
    plt.scatter(np.zeros(len(list2)) + 2, np.min(np.array(list2), -1))
    plt.scatter(np.zeros(len(list3)) + 3, np.min(np.array(list3), -1))
    plt.scatter(np.zeros(len(list4)) + 4, np.min(np.array(list4), -1))
    plt.xlabel("programme")
    plt.show()


def scatter_max(list1,list2,list3,list4):
    plt.figure(figsize=(4.5, 4.5))
    plt.title("max score of each student")
    plt.scatter(np.zeros(len(list1)) + 1, np.max(np.array(list1), -1))
    plt.scatter(np.zeros(len(list2)) + 2, np.max(np.array(list2), -1))
    plt.scatter(np.zeros(len(list3)) + 3, np.max(np.array(list3), -1))
    plt.scatter(np.zeros(len(list4)) + 4, np.max(np.array(list4), -1))
    plt.xlabel("programme")
    plt.show()


def anti_skew_data(dataframe,list1,list2,list3,list4):
    scoreMatrix = dataframe.values
    for i in range(len(scoreMatrix)):
        if scoreMatrix[i,5] == 1 and len(list1) <= 50:
            list1.append(scoreMatrix[i, :5])
        elif scoreMatrix[i,5] == 2 and len(list2) <= 50:
            list2.append(scoreMatrix[i, :5])
        elif scoreMatrix[i,5] == 3:
            list3.append(scoreMatrix[i, :5])
        elif scoreMatrix[i,5] == 4:
            list4.append(scoreMatrix[i, :5])
    transformed = np.concatenate([list1, list2, list3, list4], axis=0)
    print(transformed)


def pca(list1,list2,list3,list4):
    plt.figure(figsize=(5, 4.5))
    plt.title("PCA")
    pca = PCA(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix0)
    transformed = pca.fit_transform(matrix0)
    for i in range(len(list1)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='magenta')
    for i in range(len(list1),len(list1)+len(list2)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='blue')
    for i in range(len(list1)+len(list2),
                   len(list1)+len(list2)+len(list3)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='red')
    for i in range(len(list1)+len(list2)+len(list3),
                    len(list1) + len(list2) + len(list3) + len(list4)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='green')

    plt.show()


def ica(list1,list2,list3,list4):
    plt.figure(figsize=(5, 4.5))
    plt.title("ICA")
    ica = FastICA(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    scaler = Normalizer()
    matrix1 = scaler.fit_transform(matrix0)
    transformed = ica.fit_transform(matrix1)
    for i in range(len(list1)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='magenta')
    for i in range(len(list1), len(list1)+len(list2)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='blue')
    for i in range(len(list1)+len(list2),
                   len(list1)+len(list2)+len(list3)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='red')
    for i in range(len(list1)+len(list2)+len(list3),
                    len(list1) + len(list2) + len(list3) + len(list4)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='green')

    plt.show()


def tsne(list1,list2,list3,list4):
    plt.figure(figsize=(5,4.5))
    plt.title("TSNE")
    tsne = TSNE(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix0)
    tsne.fit_transform(matrix0)
    transformed = tsne.embedding_
    for i in range(len(list1)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='magenta')
    for i in range(len(list1), len(list1)+len(list2)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='blue')
    for i in range(len(list1)+len(list2),
                   len(list1)+len(list2)+len(list3)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='red')
    for i in range(len(list1)+len(list2)+len(list3),
                    len(list1) + len(list2) + len(list3) + len(list4)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='green')

    plt.show()


def lda(list1, list2, list3, list4):
    plt.figure(figsize=(5,4.5))
    plt.title("LDA")
    lda = LDA(n_components=2)
    matrix = np.concatenate([list1, list2, list3, list4], axis=0)
    scaler = StandardScaler()
    matrix_x = scaler.fit_transform(matrix)
    list1_y = np.zeros(len(list1)) + 1
    list2_y = np.zeros(len(list2)) + 2
    list3_y = np.zeros(len(list3)) + 3
    list4_y = np.zeros(len(list4)) + 4
    matrix_y = np.concatenate([list1_y, list2_y, list3_y, list4_y], axis=0)
    transformed = lda.fit_transform(matrix_x,matrix_y)
    for i in range(len(list1)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='magenta')
    for i in range(len(list1), len(list1)+len(list2)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='blue')
    for i in range(len(list1)+len(list2),
                   len(list1)+len(list2)+len(list3)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='red')
    for i in range(len(list1)+len(list2)+len(list3),
                    len(list1) + len(list2) + len(list3) + len(list4)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='green')

    plt.show()


def nmf(list1, list2, list3, list4):
    plt.figure(figsize=(5,4.5))
    plt.title("NMF")
    nmf = NMF(n_components=2, init='nndsvd')
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    transfer = Normalizer()
    matrix = transfer.fit_transform(matrix0)
    transformed = nmf.fit_transform(matrix)
    print(matrix0)
    for i in range(len(list1)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='magenta')
    for i in range(len(list1), len(list1)+len(list2)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='blue')
    for i in range(len(list1)+len(list2),
                   len(list1)+len(list2)+len(list3)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='red')
    for i in range(len(list1)+len(list2)+len(list3),
                    len(list1) + len(list2) + len(list3) + len(list4)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='green')
    plt.show()


def variance_filter(list1, list2, list3, list4):
    plt.figure(figsize=(5, 4.5))
    plt.title("variance filter")
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    transfer0 = VarianceThreshold(threshold=40)
    transformed = transfer0.fit_transform(matrix0)
    transfer1 = StandardScaler()
    transformed = transfer1.fit_transform(transformed)
    for i in range(len(list1)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='magenta')
    for i in range(len(list1), len(list1)+len(list2)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='blue')
    for i in range(len(list1)+len(list2),
                   len(list1)+len(list2)+len(list3)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='red')
    for i in range(len(list1)+len(list2)+len(list3),
                    len(list1) + len(list2) + len(list3) + len(list4)):
        plt.scatter(transformed[i][0], transformed[i][1], alpha=0.5, c='green')
    plt.show()


def decision_tree(list1, list2, list3, list4):
    # pca = PCA(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # transformed = pca.fit_transform(matrix0)
    # nmf = NMF(n_components=2,init='nndsvd')
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # transfer = Normalizer()
    # matrix = transfer.fit_transform(matrix0)
    # transformed = nmf.fit_transform(matrix)

    transformed = np.concatenate([list1, list2, list3, list4], axis=0)
    # dt = DecisionTreeClassifier()
    # param_dict = {"max_depth": [3,5,6,7,8,9,10,11,12,13,14,15]}
    # dt = GridSearchCV(dt, param_grid= param_dict,cv=5).fit(transformed, label)
    # print("best result using cv:", dt.best_score_)
    # print("best parameter using cv:", dt.best_params_)

    x_train, x_test, y_train, y_test = train_test_split(transformed, label, test_size=0.2)
    dt = DecisionTreeClassifier(max_depth=6)
    dt.fit(x_train, y_train)
    y_prediction1 = dt.predict(x_test)
    y_prediction2 = dt.predict(x_train)
    score1 = dt.score(x_test, y_test)
    score2 = dt.score(x_train, y_train)
    print("the prediction of test set:", y_prediction1)
    print("the accuracy: ", score1)
    print("the prediction of training set:", y_prediction2)
    print("the accuracy: ", score2)
    score = cross_val_score(dt, transformed, label, cv=5)
    print("5-fold CV score: ", score)
    warnings.filterwarnings("ignore")
    report = classification_report(y_test, y_prediction1, labels=[1, 2, 3, 4], target_names=["p1", "p2", "p3", "p4"])
    print(report)
    auc = roc_auc_score(y_test, dt.predict_proba(x_test), multi_class="ovr")
    print("auc using ovr:", auc)
    auc = roc_auc_score(y_test, dt.predict_proba(x_test), multi_class="ovo")
    print("auc using ovo:", auc)
    print("\n")

    cm = confusion_matrix(y_test, y_prediction1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def random_forest(list1, list2, list3, list4):
    # pca = PCA(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # transformed = pca.fit_transform(matrix0)
    # nmf = NMF(n_components=2,init='nndsvd')
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # transfer = Normalizer()
    # matrix = transfer.fit_transform(matrix0)
    # transformed = nmf.fit_transform(matrix)
    transformed = np.concatenate([list1, list2, list3, list4], axis=0)

    # estimator = RandomForestClassifier()
    # param_dict = {"n_estimators": [50,100,150,200,250,300]}
    # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5).fit(transformed, label)
    # print("best result using cv:", estimator.best_score_)
    # print("best parameter using cv:", estimator.best_params_)

    x_train, x_test, y_train, y_test = train_test_split(transformed, label, test_size=0.2)
    estimator = RandomForestClassifier(n_estimators=250, max_depth=6)
    estimator.fit(x_train, y_train)
    y_prediction1 = estimator.predict(x_test)
    y_prediction2 = estimator.predict(x_train)
    score1 = estimator.score(x_test, y_test)
    score2 = estimator.score(x_train, y_train)
    print("the prediction of test set:", y_prediction1)
    print("the accuracy: ", score1)
    print("the prediction of training set:", y_prediction2)
    print("the accuracy: ", score2)
    score = cross_val_score(estimator, transformed, label, cv=5)
    print("5-fold CV score: ", score)
    warnings.filterwarnings("ignore")
    report = classification_report(y_test, y_prediction1, labels=[1, 2, 3, 4], target_names=["p1", "p2", "p3", "p4"])
    print(report)
    auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class="ovr")
    print("auc using ovr:", auc)
    auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class="ovo")
    print("auc using ovo:", auc)

    cm = confusion_matrix(y_test, y_prediction1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def logistic_regression(list1, list2, list3, list4):
    # tsne = TSNE(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix0)
    # tsne.fit_transform(matrix1)
    # transformed = tsne.embedding_
    # pca = PCA(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix0)
    # transformed = pca.fit_transform(matrix1)
    matrix = np.concatenate([list1, list2, list3, list4], axis=0)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(matrix)
    # param_dict = {"solver": ['lbfgs', 'newton-cg', 'sag'], "multi_class": ['ovr', 'auto', 'multinomial']}
    # lr = LR()
    # lr = GridSearchCV(lr, param_grid=param_dict, cv=5).fit(transformed, label)
    # print("best result using cv:", lr.best_score_)
    # print("best parameter using cv:", lr.best_params_)
    # # the code following is for ignore the warning
    x_train, x_test, y_train, y_test = train_test_split(transformed, label, test_size=0.2)
    lr = LR(solver='lbfgs', multi_class='ovr')
    lr.fit(x_train, y_train)
    y_prediction1 = lr.predict(x_test)
    y_prediction2 = lr.predict(x_train)
    score1 = lr.score(x_test, y_test)
    score2 = lr.score(x_train, y_train)
    print("the prediction of test set:", y_prediction1)
    print("the accuracy: ", score1)
    print("the prediction of training set:", y_prediction2)
    print("the accuracy: ", score2)
    score = cross_val_score(lr, transformed, label, cv=5)
    print("5-fold CV score: ", score)
    warnings.filterwarnings("ignore")
    report = classification_report(y_test, y_prediction1, labels=[1, 2, 3, 4], target_names=["p1", "p2", "p3", "p4"])
    print(report)
    auc = roc_auc_score(y_test, lr.predict_proba(x_test), multi_class="ovr")
    print("auc using ovr:", auc)
    auc = roc_auc_score(y_test, lr.predict_proba(x_test), multi_class="ovo")
    print("auc using ovo:", auc)

    cm = confusion_matrix(y_test, y_prediction1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def knn(list1, list2, list3, list4):
    # tsne = TSNE(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = Normalizer()
    # matrix1 = scaler.fit_transform(matrix0)
    # tsne.fit_transform(matrix1)
    # transformed = tsne.embedding_
    # pca = PCA(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix0)
    # transformed = pca.fit_transform(matrix1)
    matrix = np.concatenate([list1, list2, list3, list4], axis=0)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(matrix)
    # estimator = KNN()
    # param_dict = {"n_neighbors": [2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], "metric": ['euclidean','manhattan',
    # 'chebyshev', 'minkowski']}
    # estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5).fit(transformed,label)
    # print("best result using cv:", estimator.best_score_)
    # print("best parameter using cv:", estimator.best_params_)

    x_train, x_test, y_train, y_test = train_test_split(transformed, label, test_size=0.2)
    estimator = KNN(n_neighbors=20, metric='chebyshev')
    estimator.fit(x_train, y_train)
    y_prediction1 = estimator.predict(x_test)
    y_prediction2 = estimator.predict(x_train)
    score1 = estimator.score(x_test, y_test)
    score2 = estimator.score(x_train, y_train)
    print("the prediction of test set:", y_prediction1)
    print("the accuracy: ", score1)
    print("the prediction of training set:", y_prediction2)
    print("the accuracy: ", score2)
    score = cross_val_score(estimator, transformed, label, cv=5)
    print("5-fold CV score: ", score)
    warnings.filterwarnings("ignore")
    report = classification_report(y_test, y_prediction1, labels=[1, 2, 3, 4], target_names=["p1", "p2", "p3", "p4"])
    print(report)
    auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class="ovr")
    print("auc using ovr:", auc)
    auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class="ovo")
    print("auc using ovo:", auc)

    # cm = confusion_matrix(y_test, y_prediction1)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()


def bayes_classifier(list1, list2, list3, list4):
    tsne = TSNE(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    scaler = StandardScaler()
    matrix1 = scaler.fit_transform(matrix0)
    tsne.fit_transform(matrix1)
    transformed = tsne.embedding_
    x_train, x_test, y_train, y_test = train_test_split(transformed, label)
    estimator = GaussianNB()
    estimator.fit(x_train, y_train)
    y_prediction1 = estimator.predict(x_test)
    y_prediction2 = estimator.predict(x_train)
    score1 = estimator.score(x_test, y_test)
    score2 = estimator.score(x_train, y_train)
    print("the prediction of test set:", y_prediction1)
    print("the accuracy: ", score1)
    print("the prediction of training set:", y_prediction2)
    print("the accuracy: ", score2)
    warnings.filterwarnings("ignore")
    report = classification_report(y_test, y_prediction1, labels=[1,2,3,4],target_names=["p1","p2","p3","p4"])
    print(report)
    auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class="ovr")
    print("auc using ovr:",auc)
    auc = roc_auc_score(y_test, estimator.predict_proba(x_test), multi_class="ovo")
    print("auc using ovo:",auc)


def svm(list1, list2, list3, list4):
    pca = PCA(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    transformed = pca.fit_transform(matrix0)
    x_train, x_test, y_train, y_test = train_test_split(transformed, label)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    param = {"kernel": ['linear', 'rbf', 'poly', 'sigmoid'], "decision_function_shape": ['ovo','ovr'], "class_weight":[None, "balanced"]}
    estimator = SVC()
    estimator = GridSearchCV(estimator, param_grid=param, cv=5).fit(x_train, y_train)
    print("best result of cv:\n", estimator.best_score_)
    print("best parameter:", estimator.best_params_)
    y_prediction = estimator.predict(x_test)
    print("the prediction value using svm:", y_prediction)
    warnings.filterwarnings("ignore")
    report = classification_report(y_test, y_prediction, labels=[1,2,3,4],target_names=["p1","p2","p3","p4"])
    print(report)


def k_means(list1, list2, list3, list4):
    # pca = PCA(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix0)
    # transformed = pca.fit_transform(matrix1)
    tsne = TSNE(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    scaler = StandardScaler()
    matrix1 = scaler.fit_transform(matrix0)
    tsne.fit_transform(matrix1)
    transformed = tsne.embedding_
    estimator = KMeans(n_clusters=4, max_iter=100).fit(transformed)
    plt.figure(figsize=(4.5,4.5))
    cluster = estimator.predict(transformed)
    plt.scatter(transformed[:,0], transformed[:,1], c=cluster)
    plt.scatter(estimator.cluster_centers_[:,0],estimator.cluster_centers_[:,1], marker="^")
    plt.show()
    sc = silhouette_score(transformed, cluster)
    print("the silhouette coefficient:", sc)


def agglomerative_clustering(list1, list2, list3, list4):
    # pca = PCA(n_components=2)
    # matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    # scaler = StandardScaler()
    # matrix1 = scaler.fit_transform(matrix0)
    # transformed = pca.fit_transform(matrix1)
    tsne = TSNE(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    scaler = StandardScaler()
    matrix1 = scaler.fit_transform(matrix0)
    tsne.fit_transform(matrix1)
    transformed = tsne.embedding_
    # ward, average, complete
    list1 = []
    list2 = [2,3,4,5,6]
    list3 = []
    for i in range(2,7):
        estimator = AgglomerativeClustering(n_clusters=i, linkage='complete')
        estimator = estimator.fit_predict(transformed)
        sc = silhouette_score(transformed, estimator)
        nmi = normalized_mutual_info_score(label, estimator)
        list1.append(sc)
        list3.append(nmi)
        plt.figure(figsize=(4.5, 4.5))
        plt.title("k = " + str(i))
        plt.scatter(transformed[:, 0], transformed[:, 1], c=estimator)
        plt.show()

    plt.figure(figsize=(4, 4.2))
    plt.title("indexes")
    plt.xlabel("x axis--value of k")
    line1, = plt.plot(list2, list1, marker='.', color='red', linestyle='--')
    line2,  = plt.plot(list2, list3, marker='.',color='blue', linestyle=':')
    plt.legend([line1, line2], ["silhouette score", "NMI score"], loc='best', frameon=False)
    plt.show()


def dbscan(list1, list2, list3, list4):
    pca = PCA(n_components=2)
    matrix0 = np.concatenate([list1, list2, list3, list4], axis=0)
    transformed = pca.fit_transform(matrix0)
    estimator = DBSCAN(eps=0.6)
    estimator = estimator.fit_predict(transformed)
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=estimator)
    plt.show()
    sc = silhouette_score(transformed, estimator)
    print("the silhouette coefficient:", sc)


if __name__ == "__main__":
    df = pd.read_csv("Data.csv", header=0)
    data_pre_processing(df)
    print_describe(df)
    correlation(df)
    data_distribution(df)
    data_skew(df)
    histogram1(df)
    histogram2(df)
    density_plot(df)
    box_whisker(df)
    correlation_matrix_plot(df)
    scatter_matrix_plot(df)
    programme1 = []
    programme2 = []
    programme3 = []
    programme4 = []
    label = []
    programme_classification(programme1, programme2, programme3, programme4, label, df)
    scatter_median(programme1, programme2, programme3, programme4)
    scatter_mean(programme1, programme2, programme3, programme4)
    scatter_std(programme1, programme2, programme3, programme4)
    scatter_min(programme1, programme2, programme3, programme4)
    scatter_max(programme1, programme2, programme3, programme4)
    pca(programme1, programme2, programme3, programme4)
    ica(programme1, programme2, programme3, programme4)
    tsne(programme1, programme2, programme3, programme4)
    lda(programme1, programme2, programme3, programme4)
    nmf(programme1, programme2, programme3, programme4)
    variance_filter(programme1, programme2, programme3, programme4)
    decision_tree(programme1, programme2, programme3, programme4)
    random_forest(programme1, programme2, programme3, programme4)
    logistic_regression(programme1, programme2, programme3, programme4)
    knn(programme1, programme2, programme3, programme4)
    bayes_classifier(programme1, programme2, programme3, programme4)
    svm(programme1, programme2, programme3, programme4)
    k_means(programme1, programme2, programme3, programme4)
    agglomerative_clustering(programme1, programme2, programme3, programme4)
    dbscan(programme1, programme2, programme3, programme4)
