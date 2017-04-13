from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from EncodeCategorical import EncodeCategorical
from PredictorUtils import PredictorUtils
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import f_classif, SelectFpr, SelectFdr, SelectKBest,SelectFwe
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.classification import accuracy_score, log_loss
from ModelComparison import ModelComparison as md
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")


filename='/Users/ridvansalih/Desktop/train_subset_001.csv'


dataset = PredictorUtils.load_data(filename)
PredictorUtils.visualize_data(dataset.rawdata)

# Construct the preprocessing pipeline
pre = Pipeline([('encoder',  EncodeCategorical(dataset.categorical_features.keys())),
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFdr(f_classif,alpha=0.05))])


# Preprocessing on pipeline
pre.fit(dataset.data, dataset.data_target)

tr_data = pre.transform(dataset.train)
val_data = pre.transform(dataset.validation)
te_data = pre.transform(dataset.test)

#Compare supervised models
md.CompareSupervisedModels(tr_data, dataset.train_target)

"""
If Grid search on MLP takes too long time, comment out it 
and use following default MLP model for supervised learning.
"""
#Grid Search on MLP model
bestmodel=md.MLPGridSearch(tr_data, dataset.train_target)
# Test the model
#bestmodel=MLPClassifier()
bestmodel.fit(tr_data, dataset.train_target)

# Use the model to get the predicted value
y_pred = bestmodel.predict(te_data)
model_probs = bestmodel.predict_proba(te_data)
y_pred_score = log_loss(dataset.test_target, model_probs[:, 1])

calibrated_model = CalibratedClassifierCV(bestmodel, method="isotonic", cv="prefit")
calibrated_model.fit(val_data, dataset.validation_target)

y_pred_cal = calibrated_model.predict(te_data)
y_pred_cal_probs = calibrated_model.predict_proba(te_data)
y_pred_cal_score = log_loss(dataset.test_target, y_pred_cal_probs[:, 1])


# execute classification report
cr1 = classification_report(dataset.test_target, y_pred, target_names=dataset.target_names)
print(cr1)
cr2 = classification_report(dataset.test_target, y_pred, target_names=dataset.target_names)
print(cr2)

print("Log-loss: %.6f (Normal Score)" % y_pred_score)
print("Log-loss: %.6f (Calibrated Score)" % y_pred_cal_score)


PredictorUtils.plot_classification_report(cr1)
PredictorUtils.plot_classification_report(cr2)

#PredictorUtils.plot_feature_relations(bestmodel,dataset.train.columns)
