import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from baselines.utils import load_data_for_baseline

pickup_path = r'E:\hx\traffic-demand-prediction\data\taxi-data\pickup-data.h5'
dropoff_path = r'E:\hx\traffic-demand-prediction\data\taxi-data\dropoff-data.h5'

data = load_data_for_baseline(pickup_path, dropoff_path, _seq_len=24 * 4, _horizon=1)
data = np.transpose(np.array(list(map(np.concatenate, data))), [0, 2, 1])
data = np.reshape(data, [-1, data.shape[-1]])

y, y_ = data[:, ].mean(axis=1), data[:, 96]

print('Test RMSE: {}'.format(mean_squared_error(y, y_) ** 0.5))
print('Test Pearson Coefficient: {}'.format(pearsonr(y, y_)))
print('Test MAE: {}'.format(mean_absolute_error(y, y_)))
