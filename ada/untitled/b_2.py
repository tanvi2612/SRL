import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

num=int(sys.argv[1])

filepath="./expected.csv"
dataset = pd.read_csv(filepath)
dropped_fields=['Label','chunk','srl','postposition','head-POS','dependency','dependency-head','predicate']
    

    #remove the current parameter from the parameters that are to be removed
    
    #for i in range(0,300):
    #    dropped_fields.append('dim_{}'.format(i))
    
    #add all the vector dimensions that must be removed
vectors=dataset.drop(dropped_fields,axis=1)
target=dataset['Label']
X_train, X_test, y_train, y_test = train_test_split(vectors, target, test_size = 0.30,stratify=target)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred=svclassifier.predict(X_test)


with open('outputs/console_head.txt','a') as f:
	print(confusion_matrix(y_test,y_pred),file=f)
	print(classification_report(y_test,y_pred),file=f)

joblib_file = "models/binary_model{}.pkl".format(num)
joblib.dump(svclassifier, joblib_file)


