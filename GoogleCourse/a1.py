import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

data_root = '.'
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

data = pickle.load(open(pickle_file, 'rb'))

plt.imshow(data['test_dataset'][1])
plt.show()

data['train_dataset'] = data['train_dataset'].reshape((data['train_dataset'].shape[0], -1))
data['test_dataset'] = data['test_dataset'].reshape((data['test_dataset'].shape[0], -1))

logistic = LogisticRegression()
print("Training")
train = logistic.fit(data['train_dataset'], data['train_labels'])
print('LogisticRegression score: %f' % train.score(data['test_dataset'], data['test_labels']))

print("Predictions:", logistic.predict(data['test_dataset'][1]))