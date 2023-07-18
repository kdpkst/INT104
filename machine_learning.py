import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import load_iris,load_boston, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#  LinearRegression use normal equation while SGDRegressor use SGD gradient descent, ridge is linear regression with
#  regularization
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score, silhouette_score
import joblib
from sklearn.cluster import KMeans

def dataset_demo():
    iris = load_iris()
    irisData = iris.data
    irisTarget = iris.target
    # split the data into training set and testing set
    # four parameters: features, labels, size of testing set and random state
    x_train, x_test, y_train, y_test = train_test_split(irisData, irisTarget, test_size=0.2)

def dict_feature_extraction():
    # the data below is just an example:
    data = [{'name': 'curry', 'three': 405},
            {'name': 'durant', 'three': 254},
            {'name': 'harden', 'three': 384},
            {'name': 'thompson', 'three': 354}]
# sparse matrix: only represent non-zero values, which saves space and improve efficiency
# the default value of DictVectorizer is "sparse=True"
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names())
    print(data_new)

def count_demo():
    # the data below is just an example:
    data = ["i really really love python", "but python hates me"]
    # this method can extract single English word as the feature, but it cannot work when dealing with Chinese
    # if hoping to extract features from Chinese texts, separate chinese characters first.
    # transfer = CountVectorizer(stop_words = [...]) can be used.
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    # the "toarray" here is a function which helps print out "data_new" in the form of non-sparse array
    print(transfer.get_feature_names())
    print(data_new.toarray())

def tfidf():
    # the data below is just an example:
    data = ["Machine learning is a branch of artificial intelligence and computer science",
            "which focuses on the use of data and algorithms to imitate the way",
            "that humans learn",
            "gradually improving its accuracy"]
    transfer = TfidfVectorizer(stop_words=['the', 'of', 'is', 'that'])
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names())
    print(data_new.toarray())

def min_max_scaler():
     data = pd.read_csv("CW_Data.csv", header=0)
     data = data.iloc[:, 1:6]
     # can also apply this: transfer = MinMaxScaler(feature_range=[x,y])
     transfer = MinMaxScaler()
     data_new = transfer.fit_transform(data)
     print(data_new)

def standard_scaler():
    data = pd.read_csv("CW_Data.csv", header=0)
    data = data.iloc[:, 1:6]
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)


def variance_filter():
    data = pd.read_csv("CW_Data.csv", header=0)
    data = data.iloc[:, 1:6]
    # transfer = VarianceThreshold(threshold=x), can also set parameter
    transfer = VarianceThreshold(threshold=40)
    data_new = transfer.fit_transform(data)
    print(data_new,data_new.shape)

def pearson():
    data = pd.read_csv("CW_Data.csv", header=0)
    data = data.iloc[:, 1:6]
    # define a variable "r" to store the value of pearson correlation coefficient
    # the code below is just an example
    r = pearsonr(data["Q3"],data["Q5"])
    # a tuple will be printed out and the first value of the tuple is the correlation coefficient "r"
    print(r)

def pca_demo():
    # the data below is just an example
    data = [[1,3,5,2],[3,1,88,3],[9,0,45,7]]
    # if the value of "n_components" is a decimal number, which means differently
    # eg. n_components=0.95, which means preserve 95% of the original information
    pca = PCA(n_components=2)
    data_new = pca.fit_transform(data)
    print(data_new)

def knn_iris():
    iris = load_iris()
    # a parameter is "random_state".
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    # data preprocessing, note the way to preprocess x_test(only use "transform")
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # model the knn algorithm
    estimator = KNN(n_neighbors=7)
    estimator.fit(x_train, y_train)
    # estimate this model
    y_prediction = estimator.predict(x_test)
    # compare the real value and the prediction value
    print(y_prediction)
    print(y_prediction == y_test)
    # calculate the precision rate
    rate = estimator.score(x_test, y_test)
    print(rate)

def knn_iris_gscv():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
    # data preprocessing, note the way to preprocess x_test(only use "transform")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # model the knn algorithm
    # since we will optimize the parameter "k", there is no need to input a parameter "k"
    estimator = KNN()
    # use grid search and cross validation
    param_dict = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    # input the three relatively important parameters as follows:
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10).fit(x_train, y_train)
    score = estimator.score(x_test,y_test)
    print("the accuracy rate: ", score)
    # use the functions to analyze the result
    print("best parameter:\n", estimator.best_params_)
    print("best result:\n", estimator.best_score_)
    print("best estimator:\n", estimator.best_estimator_)
    print("result of cross validation:\n", estimator.cv_results_)

def naive_bayes_20news():
    # acquire the data from "fetch_20newsgroups"
    news = fetch_20newsgroups(subset="all")
    # split the training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    # extract the text features
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # the default value of parameter alpha is set to be 1(it can be changed if want)
    nb = MultinomialNB()
    nb.fit(x_train,y_train)
    y_prediction = nb.predict(x_test)
    print(y_prediction)
    score = nb.score(x_test,y_test)
    print(score)

def decision_tree_iris():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# it is not necessary to use "standardscaler" for the data in this case as the decision tree is not knn which need to
# compute the distance.
# the parameter "criterion" is the criterion used by the decision tree to build the mathematical model, the default
# criterion is "gini"
# the other parameter "max_depth": if this parameter is too big, which may lead to the overfitting problem
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    dt.fit(x_train, y_train)
    y_prediction = dt.predict(x_test)
    score = dt.score(x_test, y_test)
    print(y_prediction)
    print(score)

def random_forest_iris_gscv():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# the parameter in random_forest "n_estimator" means n decision trees
    param_dict = {"n_estimators": [3, 5, 7, 9, 11],"max_depth":[5,7,9,11,13]}
    estimator = RandomForestClassifier()
    # input the three relatively important parameters as follows:
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10).fit(x_train, y_train)
    score = estimator.score(x_test,y_test)
    print(score)
    print("best result:\n", estimator.best_score_)
    print("best estimator:\n", estimator.best_estimator_)

def linear_normal_equation():
    # load the dataset
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target)
    # data preprocessing
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # build the linear regression model(normal equation)
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    print("normal equation--coefficients:", estimator.coef_)
    print("normal equation--intercept:", estimator.intercept_)
    # evaluate the linear model
    y_prediction = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_prediction)
    print("normal equation mean squared error:", error)


def linear_gradient_descent():
    # load the dataset
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target)
    # data preprocessing
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # build the linear regression model(gradient descent), there are some parameters in SGDRegressor
    # if you do not set ant parameters(that's ok), the default values or algorithms will be used
    # parameters contain learning rate, whether setting a intercept......
    estimator = SGDRegressor(learning_rate="constant",eta0=0.01)
    estimator.fit(x_train, y_train)
    print("gradient descent--coefficients:", estimator.coef_)
    print("gradient descent--intercept:", estimator.intercept_)
    # evaluate the linear model
    y_prediction = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_prediction)
    print("gradient descent mean squared error:", error)

def ridge_regression():
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # SGDRegressor() parameter can set ridge regression
    # Ridge() has a parameter "alpha" which can be set to adjust the degree of penalty
    estimator = Ridge(alpha=0.01)
    estimator.fit(x_train, y_train)
    print("ridge regression--coefficients:", estimator.coef_)
    print("ridge regression--intercept:", estimator.intercept_)
    y_prediction = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_prediction)
    print("ridge regression mean squared error:", error)

def logistic_regression():
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print("the coefficients of logistic regression: ", lr.coef_)
    print("the intercept of logistic regression", lr.intercept_)
    score = lr.score(x_test, y_test)
    print("the accuracy rate of logistic regression:", score)
    y_prediction = lr.predict(x_test)
    report = classification_report(y_test, y_prediction, labels=[1,0],target_names=["malignant","benign"])
    print(report)
    auc = roc_auc_score(y_test, y_prediction)
    print("auc:",auc)

def save_model():
    # the example below shows the way to save the model
    # load the dataset
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target)
    # data preprocessing
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # build the linear regression model(gradient descent), there are some parameters in SGDRegressor
    # if you do not set ant parameters(that's ok), the default values or algorithms will be used
    # parameters contain learning rate, whether setting a intercept......
    # estimator = SGDRegressor(learning_rate="constant",eta0=0.01)
    # estimator.fit(x_train, y_train)
    # use function joblib.dump to save the model, example below:
    # joblib.dump(estimator, "sgdLineaerRegression.pkl")
    # use function joblib.load to re_load the saved model, example below:
    # estimator = joblib.load("sgdLineaerRegression.pkl")

def k_means():
    iris = load_iris()
# the grid search can be used to find the value of best parameter "n_clustering"
    estimator = KMeans(n_clusters=3)
# it is an unsupervised learning algorithm, which only the features need to be the input as parameter
    estimator.fit(iris.data)
# estimator.predict(features) is used to check the result of the clustering
    cluster = estimator.predict(iris.data)
    print("the clustering: \n",cluster)
# apply the silhouette coefficient to evaluate the k-means model
    sc = silhouette_score(iris.data, cluster)
    print("the silhouette coefficient:",sc)

def moon_dataset():
    plt.figure()
    x, y = make_moons(n_samples=200, noise=0.009)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

def main():
    # sklearn dataset usage (iris dataset)
    dataset_demo()
    # dictionary features extraction using sklearn--DictVectorizer
    # dict_feature_extraction()
    # English texts features extraction using sklearn--CountVectorizer
    # count_demo()
    # use tfidf to extract features from texts--TfidfVectorizer
    # tfidf()
    # data preprocessing
    # min_max_scaler()
    # standard_scaler()
    # delete the features which have very low variance
    # variance_filter()
    # calculate the value of pearson correlation coefficient
    # pearson()
    # pca_demo()
    # knn_iris()
    # knn_iris_gscv()
    # naive_bayes_20news()
    # decision_tree_iris()
    # random_forest_iris_gscv()
    # linear_normal_equation()
    # linear_gradient_descent()
    # ridge_regression()
    # logistic_regression()
    # save_model()
    # k_means()
    # moon_dataset()


if __name__ == "__main__":
    main()
