import numpy as np
from joblib import dump, load
import lightgbm as lgb
from sklearn.model_selection import train_test_split

data_path = '/home/yessense/PycharmProjects/RoboDogControl/data/data.npy'
labels_path = '/home/yessense/PycharmProjects/RoboDogControl/data/labels.npy'
classifier_path = '/home/yessense/PycharmProjects/RoboDogControl/data/classifier.joblib'

data = np.load(data_path)
data = data.reshape(-1, 33 * 4)
labels = np.load(labels_path)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=0)

classifier = lgb.LGBMClassifier(random_state=0)
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)

dump(classifier, classifier_path)
classifier = load(classifier_path)
score = classifier.score(X_test, y_test)
print(score)
print("done")
