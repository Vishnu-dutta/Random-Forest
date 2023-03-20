import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sn

digits = load_digits()
plt.gray()
# for i in range(4):
#     plt.imshow(digits.images[i])
#     plt.show()

df = pd.DataFrame(digits.data)
df['target'] = digits.target

X = df.drop('target', axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()