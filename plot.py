import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

dirs = {'xgboost-taxi': r'E:\qyn\traffic-network\data\model\xgboost-taxi',
        'lstm-taxi': r'data/model/lstm-taxi',
        'dgcgru-taxi-3': r'data/model/dgcgru-taxi-3',
        'dgcrn-taxi': r'data/model/dgcrn-taxi'}

# dirs = {'xgboost-bike': r'E:\qyn\traffic-network\data\model\xgboost-bike',
#         'lstm-bike-3': r'data/model/lstm-bike-3',
#         'dgcgru-bike-3': r'data/model/dgcgru-bike-3',
#         'dgcrn-bike': r'data/model/dgcrn-bike'}

# data_names = ['{}-result-7.npz', '{}-result-8.npz', '{}-result-9.npz']
# data_names = ['{}-result-17.npz', '{}-result-18.npz', '{}-result-19.npz']
# data_names = ['{}-result-12.npz', '{}-result-13.npz', '{}-result-14.npz']
data_names = ['{}-result-18.npz', '{}-result-19.npz', '{}-result-20.npz']

rmses, pccs, maes = list(), list(), list()
for method, dir in dirs.items():
    labels, predictions = list(), list()
    for name in data_names:
        data = np.load(os.path.join(dir, name.format(method)))
        label, prediction = data['labels'], data['predictions']
        labels.append(label)
        predictions.append(prediction)
    labels = np.concatenate(labels, axis=0).flatten()
    predictions = np.concatenate(predictions, axis=0).flatten()

    rmses.append(mean_squared_error(labels, predictions) ** .5)
    pcc, _ = pearsonr(labels, predictions)
    pccs.append(pcc)
    maes.append(mean_absolute_error(labels, predictions))

data = pd.DataFrame([rmses, maes], index=['rmse', 'mae'], columns=['xgboost', 'lstm', 'dgcgru', 'dgcrn'])
# sns.barplot(x=data.transpose().index, y=data.transpose().values)
data.transpose().plot.bar(color=['#069CCF', '#A5BE6A', '#EEC900'])
plt.xticks(np.arange(4), ['xgboost', 'lstm', 'dgcgru', 'dgcrn'], rotation=360)
plt.show()
##################################################################################################################


dir = r'data/model/dgcrn-bike'
rmses, pccs, maes = dict(), dict(), dict()
for hour in range(24):
    data = np.load(os.path.join(dir, 'dgcrn-bike-result-{}.npz'.format(hour)))
    label, prediction = data['labels'].flatten(), data['predictions'].flatten()
    rmses[hour] = mean_squared_error(label, prediction) ** .5
    pccs[hour], _ = pearsonr(label, prediction)
    maes[hour] = mean_absolute_error(label, prediction)

data = pd.DataFrame([rmses, pccs, maes], index=['rmse', 'pcc', 'mae'])
data.transpose().plot(kind='bar')
import matplotlib.pyplot as plt

plt.show()
