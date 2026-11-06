import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix)
from sklearn.datasets import load_iris

data=load_iris()

df=pd.DataFrame(data.data,columns=data.feature_names)
df["Result"]=data.target

X=df.drop('Result',axis=1)
y=df['Result']

print(df.head())
print(df.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted',zero_division=0)
recall=recall_score(y_test,y_pred,average='weighted',zero_division=0)
f1_cls=f1_score(y_test,y_pred,average='weighted',zero_division=0)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)

print(accuracy)
print(precision)
print(recall)
print(f1_cls)
print(cm)
print(cr)