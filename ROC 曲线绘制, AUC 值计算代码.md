## AUC_score 计算
Examples
### 
### Binary case:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
roc_auc_score(y, clf.predict_proba(X)[:, 1])

roc_auc_score(y, clf.decision_function(X))
```
0.9945298874266688
### Multiclass case:
```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(solver="liblinear").fit(X, y)
roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
```
0.9913333333333334
### Multilabel case:
```python
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
X, y = make_multilabel_classification(random_state=0)
clf = MultiOutputClassifier(clf).fit(X, y)
# get a list of n_output containing probability arrays of shape
# (n_samples, n_classes)
y_pred = clf.predict_proba(X)
# extract the positive columns for each output
y_pred = np.transpose([pred[:, 1] for pred in y_pred])
roc_auc_score(y, y_pred, average=None)

from sklearn.linear_model import RidgeClassifierCV
clf = RidgeClassifierCV().fit(X, y)
roc_auc_score(y, clf.decision_function(X), average=None)
```
[0.81996435 0.8467387  0.93090909 0.87229702 0.94422994]
## ROC 曲线绘制
**Examples**
```python
import matplotlib.pyplot as plt  
from sklearn import datasets, metrics, model_selection, svm
X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(X_train, y_train)

metrics.plot_roc_curve(clf, X_test, y_test)  
plt.show()  
```
![image.png](https://cdn.nlark.com/yuque/0/2021/png/2683368/1615429291955-cde657c1-e70d-4378-8b3e-751a8f135edb.png#align=left&display=inline&height=583&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1166&originWidth=1280&size=92508&status=done&style=none&width=640)
## Roc曲线显示
```python
import matplotlib.pyplot as plt  
import numpy as np
from sklearn import metrics
y = np.array([0, 0, 1, 1])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,                                          estimator_name='example estimator')
display.plot()  
plt.show()
```
![image.png](https://cdn.nlark.com/yuque/0/2021/png/2683368/1615429533397-847d4b00-f765-44fe-852b-8f361cfc2e29.png#align=left&display=inline&height=583&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1166&originWidth=1280&size=86960&status=done&style=none&width=640)
## 平均精度分数
average precision score
![image.png](https://cdn.nlark.com/yuque/0/2021/png/2683368/1615429915511-7d241014-38b8-4e3f-9acf-cb2faeaf61de.png#align=left&display=inline&height=74&margin=%5Bobject%20Object%5D&name=image.png&originHeight=276&originWidth=1390&size=34138&status=done&style=none&width=374)
![image.png](https://cdn.nlark.com/yuque/0/2021/png/2683368/1615429899908-a80ab2ce-f281-47b4-97f7-3129c297e97e.png#align=left&display=inline&height=138&margin=%5Bobject%20Object%5D&name=image.png&originHeight=276&originWidth=1390&size=34138&status=done&style=none&width=695)![image.png](https://cdn.nlark.com/yuque/0/2021/png/2683368/1615429902699-d76860e8-c0dd-45b4-b324-7bfaea46e962.png#align=left&display=inline&height=138&margin=%5Bobject%20Object%5D&name=image.png&originHeight=276&originWidth=1390&size=34138&status=done&style=none&width=695)
```python
import numpy as np
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
average_precision_score(y_true, y_scores)
```
0.8333333333333333
## precision recall curve
```python
import numpy as np
from sklearn.metrics import precision_recall_curve
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(
    y_true, y_scores)
print(precision)

print(recall)

print(thresholds)
```
[0.66666667 0.5        1.         1.        ]
[1.  0.5 0.5 0. ]
[0.35 0.4  0.8 ]
## PrecisionRecallDisplay
```python
from sklearn.datasets import make_classification
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
precision, recall, _ = precision_recall_curve(y_test, predictions)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot() 
```
## 检测误差权衡(DET)曲线
```python
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_det_curve
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

N_SAMPLES = 1000

classifiers = {
    "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
    "Random Forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1
    ),
}

X, y = make_classification(
    n_samples=N_SAMPLES, n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.4, random_state=0)

# prepare plots
fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)

    plot_roc_curve(clf, X_test, y_test, ax=ax_roc, name=name)
    plot_det_curve(clf, X_test, y_test, ax=ax_det, name=name)

ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')
ax_det.set_title('Detection Error Tradeoff (DET) curves')

ax_roc.grid(linestyle='--')
ax_det.grid(linestyle='--')

plt.legend()
plt.show()
```
![image.png](https://cdn.nlark.com/yuque/0/2021/png/2683368/1615429824526-8c4f026f-2cc6-4fa3-abcb-bc4771405363.png#align=left&display=inline&height=603&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1206&originWidth=2200&size=228784&status=done&style=none&width=1100)
