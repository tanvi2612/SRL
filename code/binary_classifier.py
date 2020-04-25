import pandas as pd
import numpy as np

filepath="./out.csv"
dataset = pd.read_csv(filepath)
vectors=dataset.drop('Labels',axis=1)
target=dataset['Labels']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, target, test_size = 0.30)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


from sklearn.externals import joblib

joblib_file = "binary_model.pkl"
joblib.dump(svclassifier, joblib_file)

# Load from file
# joblib_model = joblib.load(joblib_file)
