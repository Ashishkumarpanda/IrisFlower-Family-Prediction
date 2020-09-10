import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
data=pd.read_csv("Iris.csv")
#finding unique values in dependent column

data['Species'].unique()
#converting string to numerical value

data['Species'].replace('Iris-setosa',0,inplace=True)
data['Species'].replace('Iris-versicolor',1,inplace=True)
data['Species'].replace('Iris-virginica',2,inplace=True)
data.head()
x=data.iloc[:,0:5].values
y=data.iloc[:,5].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)
prediction_y=classifier.predict(x_test)
accuracy=metrics.accuracy_score(y_test,prediction_y)
value=accuracy*100
print("accuracy is=",str(value)+" %")
plt.plot(x_train,classifier.predict(x_train),'o-')
plt.xlabel("Iris FLower Details")
plt.ylabel("Flower Family")
label=['Iris-setosa','Iris-versicolor','Iris-virginica']
plt.legend(label)
plt.show()









