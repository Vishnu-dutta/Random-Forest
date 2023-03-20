from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), df['target'], test_size=0.2)
'''
by default n_estimators here is 100
'''
reg = RandomForestClassifier(n_estimators=10)
reg.fit(X_train,y_train)
print(reg.score(X_test,y_test))
