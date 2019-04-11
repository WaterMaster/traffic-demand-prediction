import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm

from baselines.utils import load_data_for_baseline

seq_len = 800
pickup_path = r'E:\hx\traffic-demand-prediction\data\bike-data\pickup-data.h5'
dropoff_path = r'E:\hx\traffic-demand-prediction\data\bike-data\dropoff-data.h5'

data = load_data_for_baseline(pickup_path, dropoff_path, _seq_len=seq_len, _horizon=1)
labels, predictions = list(), list()
for (xs, ys) in tqdm(data):
    try:
        model = VAR(xs)
        results = model.fit(trend='ctt')
        lag_order = results.k_ar
        prediction = results.forecast(xs[-lag_order:], len(ys))

        labels.append(ys[0])
        predictions.append(prediction[0])
    except:
        pass
y_true = np.concatenate(labels)
y_pred = np.concatenate(predictions)
print('Test RMSE: {}'.format(mean_squared_error(y_true, y_pred) ** 0.5))
print('Test Pearson Coefficient: {}'.format(pearsonr(y_true, y_pred)))
print('Test MAE: {}'.format(mean_absolute_error(y_true, y_pred)))
