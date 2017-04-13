import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.datasets.base import Bunch
import sklearn.cross_validation as cross_validation
from sklearn.preprocessing import LabelEncoder


class PredictorUtils:

    @staticmethod
    def load_data(filename,size=None):
        """
        It loads the data from file and parse into the features.
        It returns the bunch of train, test and validation data with their target values. 
        """

        names = [
            'id',
            'click',
            'hour',
            'C1',
            'banner_pos',
            'site_id',
            'site_domain',
            'site_category',
            'app_id',
            'app_domain',
            'app_category',
            'device_id',
            'device_ip',
            'device_model',
            'device_type',
            'device_conn_type',
            'C14',
            'C15',
            'C16',
            'C17',
            'C18',
            'C19',
            'C20',
            'C21',
        ]
        arr=list(range(1, 24))

        rawdata = pd.read_csv(filename, sep="\s*,", engine='python',names=names,skiprows=[0],usecols=arr,nrows=size)

        rawdata=rawdata.astype(str)
        rawdata['day'] = rawdata['hour'].str[4:-2]
        rawdata['hour'] = rawdata['hour'].str[-2:]


        print(rawdata.describe())

        #cols = ["hour"]+["C1"]+["banner_pos"]+["site_category"]+\
        #       ["app_domain"]+["app_category"]+["device_type"]+\
        #       ["device_conn_type"]+["C15"]+["C16"]+["C18"]
        #binary_data = pd.get_dummies(rawdata,columns=cols)


        meta = {
            'target_names': list(rawdata.click.unique()),
            'feature_names': list(rawdata.columns),
            'categorical_features': {
                column: list(rawdata[column].unique())
                for column in rawdata.columns
                if rawdata[column].dtype == 'object'
            },
        }

        names = meta['feature_names']
        meta['categorical_features'].pop('click')

        train, val = cross_validation.train_test_split(rawdata, test_size=0.30)
        validation, test = cross_validation.train_test_split(val, test_size=0.50)

        y_encode = LabelEncoder().fit(train['click'])

        # Return the bunch with the appropriate data chunked apart
        return Bunch(
            rawdata=rawdata,
            data=rawdata.drop('click', axis=1),
            data_target=rawdata['click'],
            train=train.drop('click', axis=1),
            train_target=y_encode.transform(train['click']),
            validation=validation.drop('click', axis=1),
            validation_target=y_encode.transform(validation['click']),
            test=test.drop('click', axis=1),
            test_target=y_encode.transform(test['click']),
            target_names=meta['target_names'],
            feature_names=meta['feature_names'],
            categorical_features=meta['categorical_features'],
        )

    @staticmethod
    def visualize_data(data):
        """
        It is used to plot data and observe some relations among features.
        """

        encoded_data, _ = PredictorUtils.number_encode_features(data)
        sns.heatmap(encoded_data.corr(), square=True)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()

        sns.countplot(y='hour', hue='click', data=encoded_data)
        sns.plt.title('hours vs click')
        sns.plt.show()

        sns.countplot(x='site_category', hue='click', data=encoded_data)
        sns.plt.title('site_category vs click')
        sns.plt.show()

        sns.countplot(x='app_category', hue='click', data=encoded_data)
        sns.plt.title('app_category vs click')
        sns.plt.show()


        g = sns.FacetGrid(encoded_data, col='device_type', size=4, aspect=.5)
        g = g.map(sns.boxplot, 'click', 'hour')
        sns.plt.show()

        sns.violinplot(x='day', y='hour', hue='click', data=encoded_data, split=True, scale='count')
        sns.plt.title('Hour and Day vs Click')
        sns.plt.show()

        g = sns.PairGrid(encoded_data,
                         x_vars=['device_type'],
                         y_vars=['hour'],
                         aspect=.75, size=3.5)
        g.map(sns.violinplot, palette='pastel')
        sns.plt.show()

        g = sns.PairGrid(encoded_data,
                         x_vars=['device_type'],
                         y_vars=['day'],
                         aspect=.75, size=3.5)
        g.map(sns.violinplot, palette='pastel')
        sns.plt.show()


    @staticmethod
    def plot_classification_report(cr, title=None, cmap=cm.YlOrRd):
        """
        It is used to visualize classification results. 
        """
        title = title or 'Classification report'
        lines = cr.split('\n')
        classes = []
        matrix = []

        for line in lines[2:(len(lines) - 3)]:
            s = line.split()
            classes.append(s[0])
            value = [float(x) for x in s[1: len(s) - 1]]
            matrix.append(value)

        fig, ax = plt.subplots(1)

        for column in range(len(matrix) + 1):
            for row in range(len(classes)):
                txt = matrix[row][column]
                ax.text(column, row, matrix[row][column], va='center', ha='center')

        fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(len(classes) + 1)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.ylabel('Classes')
        plt.xlabel('Measures')
        plt.show()

    @staticmethod
    def plot_feature_relations(classifier, columns):
        """
        Positive or negative effects of features on the model can be visualized 
         by using this function. However, classifier should have "coef_" in order 
         to plot the relation. 
        """
        coefs = pd.Series(classifier.coef_[0], index=columns)
        coefs.sort()
        coefs.plot(kind="bar")
        plt.show()

    @staticmethod
    def number_encode_features(df):
        result = df.copy()
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column])
        return result, encoders


