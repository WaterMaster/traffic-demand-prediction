import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AR
from tqdm import tqdm

from baselines.utils import load_data_for_baseline

pickup_path = r'E:\hx\traffic-demand-prediction\data\taxi-data\pickup-data.h5'
dropoff_path = r'E:\hx\traffic-demand-prediction\data\taxi-data\dropoff-data.h5'

seq_len = 80
data = load_data_for_baseline(pickup_path, dropoff_path, _seq_len=seq_len, _horizon=1)
data = np.transpose(np.array(list(map(np.concatenate, data))), [0, 2, 1])
data = np.reshape(data, [-1, data.shape[-1]])

# data = np.array(random.sample(list(data), 100000))

predictions = list()
trains, labels = np.split(data, [seq_len], axis=1)
for x in tqdm(trains):
    model = AR(x)
    model_fit = model.fit()
    model_fit.k_ar = 24
    prediction = model_fit.predict(start=len(x), end=len(x), dynamic=True)[0]
    predictions.append(prediction)
y, y_ = labels[:, 0], np.array(predictions)
print('Test RMSE: {}'.format(mean_squared_error(y, y_) ** 0.5))
print('Test Pearson Coefficient: {}'.format(pearsonr(y, y_)))
print('Test MAE: {}'.format(mean_absolute_error(y, y_)))

# # plot results
# plt.figure()
# plt.title('ARM Predictions & Labels')
# plt.scatter(labels, np.round(predictions), s=3, c='r', alpha=0.2)
# plt.xlim(0, 200)
# plt.ylim(0, 200)
# plt.xlabel('Labels')
# plt.ylabel('Predictions')
# plt.show()
