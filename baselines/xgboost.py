import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from baselines.utils import load_data_for_baseline

pickup_path = 'data/taxi-data/pickup-data.h5'
dropoff_path = 'data/taxi-data/dropoff-data.h5'
train_partition = 0.7
seq_len = 80

data = load_data_for_baseline(pickup_path, dropoff_path, _seq_len=seq_len)

train = data[:int(train_partition * len(data))]
tests = data[int(train_partition * len(data)):]

train = np.transpose(np.array(list(map(np.concatenate, train))), [0, 2, 1])
train = np.reshape(train, [-1, train.shape[-1]])


x, y = np.split(train, [seq_len], axis=1)

model = xgb.XGBRegressor(learning_rate=0.1, max_depth=15, gamma=1e-3, n_estimators=10, objective='reg:linear')
# train model
model.fit(x, y)
# test
rmses, pccs = list(), list()
for test in tests:
    test = np.transpose(np.concatenate(test))
    test_inputs, labels = np.split(test, [seq_len], axis=1)
    predictions = model.predict(test_inputs)
    labels = np.transpose(labels)[0]

    rmses.append(mean_squared_error(labels, predictions) ** 0.5)
    pr, p = pearsonr(predictions, labels)
    pccs.append(pr)
print('Test RMSE: {}'.format(np.mean(rmses)))
print('Test Pearson Coefficient: {}'.format(np.mean(pccs)))