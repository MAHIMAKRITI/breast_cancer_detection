import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#loading the dataset
df = pd.read_csv('C:/Users/HP/Downloads/data (1).csv')
pd.set_option('display.expand_frame_repr', False)
df = df.iloc[:,1:-1]

#peek into the dataset
#print(df.info())
#print(df.head(5))


X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
#print(accuracy)


#converting categorical data into nummerical values
df['diagnosis'] =df['diagnosis'].astype('category').cat.codes


label_encoder = LabelEncoder()
df.iloc[:,0] = label_encoder.fit_transform(df.iloc[:,0]).astype('float64')


#making a heat map to show correlation between the different features
ax = sns.heatmap(df[df.columns[0:31]].corr(), annot = True)
plt.show()
#eliminating one of the two features with correlation greater then 0.9
corr = df.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]
X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
