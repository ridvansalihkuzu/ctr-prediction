import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")

class ModelComparison:


    @staticmethod
    def CompareSupervisedModels(data,target):
        """
        Compares model performances of Random Forest, Multilayer Perceptron, 
        Logistic Regression, Linear Discriminant Analysis, KNN, Decision Tree, 
        and Naïve Bayes by using thier default parameters.
        """
        models = []
        results = []
        names = []

        print("BEST MODEL SEARCH:")
        models.append(('RFC', RandomForestClassifier()))
        models.append(('MLP', MLPClassifier()))
        models.append(('LRC', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DTC', DecisionTreeClassifier()))
        models.append(('NBC', GaussianNB()))


        # evaluate each model in turn
        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, random_state=7)
            cv_results = model_selection.cross_val_score(model, data,target, cv=kfold, scoring='neg_log_loss')
            results.append(cv_results)
            names.append(name)
            print("MODEL: {}, ACCURACY: {:.6f} (+/-{:.6f})"
                  .format(name, cv_results.mean(), cv_results.std() / 2))

        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

    @staticmethod
    def MLPGridSearch(data,target):
        """
        Grid search on hyper-parameters of Multilayer Perceptron. 
        It returns the model with the best parameters.
        """
        print("GRID SEARCH on MLP:")
        parameters = [{'hidden_layer_sizes': [100,200,300],
                       'solver': ['sgd', 'adam'],
                       'alpha': [0.0001, 0.001, 0.01],
                       'activation': ['relu', 'tanh']},]
        clf = GridSearchCV(MLPClassifier(hidden_layer_sizes=1, activation=1, solver=1,alpha=1),
                           parameters, cv=5, scoring='neg_log_loss')

        clf.fit(data,target)
        for params, mean_score, scores in clf.grid_scores_:
            print("{}: {:.6f} (+/-{:.6f})".format(params, mean_score, scores.std() / 2))

        print("The best model for MLP has hidden_layer_sizes={}, activation={}, solver={}, alpha={},  SCORE={:.6f}"
                  .format(clf.best_estimator_.hidden_layer_sizes,clf.best_estimator_.activation,
                          clf.best_estimator_.solver, clf.best_estimator_.alpha,
                          clf.best_score_))

        return MLPClassifier(hidden_layer_sizes=clf.best_estimator_.hidden_layer_sizes,
                             solver=clf.best_estimator_.solver,
                             alpha=clf.best_estimator_.alpha,
                             activation=clf.best_estimator_.activation)

    @staticmethod
    def LogisticRegressionGridSearch(data,target):
        """
        Grid search on hyper-parameters of Logistic Regression. 
        It returns the model with the best parameters.
        """
        print("GRID SEARCH on Logistic Regression:")
        parameters = [{'C': [0.01, 0.05, 0.1, 0.5, 1],
                       'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag']},]
        clf = GridSearchCV(LogisticRegression(C=1, solver=1),
                           parameters, cv=5, scoring='neg_log_loss')

        clf.fit(data,target)
        for params, mean_score, scores in clf.grid_scores_:
            print("{}: {:.6f} (+/-{:.6f})".format(params, mean_score, scores.std() / 2))

        print("The best model for Logistic Regression has C={}, solver={}, SCORE={:.6f}"
                  .format(clf.best_estimator_.C,clf.best_estimator_.solver,
                          clf.best_score_))

        return LogisticRegression(C=clf.best_estimator_.C,
                             solver=clf.best_estimator_.solver)
