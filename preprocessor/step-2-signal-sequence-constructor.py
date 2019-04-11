import pandas as pd
import numpy as np
from sklearn import preprocessing


def preprocess_taxi_data(taxi_data_path, cluster_result):
    taxi_data: pd.DataFrame = pd.read_hdf(taxi_data_path, key='taxi')
    cluster_result: pd.DataFrame = pd.read_hdf(cluster_result, key='taxi')

    taxi_data['pickup_cluster_no'] = taxi_data.apply(
        lambda row: cluster_result.at[int(row['pickup_unit_no']), 'belong'],
        axis=1)
    taxi_data['dropoff_cluster_no'] = taxi_data.apply(
        lambda row: cluster_result.at[int(row['dropoff_unit_no']), 'belong'],
        axis=1)
    taxi_data.to_hdf(taxi_data_path, key='taxi')
    taxi_data = taxi_data.drop(columns=['pickup_unit_no', 'dropoff_unit_no'])
    return taxi_data


def signal_sequence_construct(_data, _node_list, key, pickup_path, dropoff_path, freq):
    """
    construct signal sequence from data
    :return: two pd.DataFrame, with node as columns and time as indexes
    """
    count_by_pickup_time = _data.groupby([pd.Grouper(key='pickup_datetime', freq=freq), 'pickup_cluster_no']).size()
    count_by_dropoff_time = _data.groupby(
        [pd.Grouper(key='dropoff_datetime', freq=freq), 'dropoff_cluster_no']).size()

    _pickup_signal_sequence: pd.DataFrame = count_by_pickup_time.unstack(level='pickup_cluster_no', fill_value=0)
    _dropoff_signal_sequence: pd.DataFrame = count_by_dropoff_time.unstack(level='dropoff_cluster_no', fill_value=0)

    times = np.union1d(_pickup_signal_sequence.index.values, _dropoff_signal_sequence.index.values)

    _pickup_signal_sequence = _pickup_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)
    _dropoff_signal_sequence = _dropoff_signal_sequence.reindex(index=times, columns=_node_list, fill_value=0)

    _pickup_signal_sequence.to_hdf(pickup_path, key=key)
    _dropoff_signal_sequence.to_hdf(dropoff_path, key=key)

    return _pickup_signal_sequence, _dropoff_signal_sequence


# def graph_sequence_construct(_pickup_signal_sequence, _dropoff_signal_sequence):
#     pickup_by_hour = _pickup_signal_sequence.groupby([(_pickup_signal_sequence.index.hour // time_frame)]).sum()
#     dropoff_by_hour = _dropoff_signal_sequence.groupby([(_dropoff_signal_sequence.index.hour // time_frame)]).sum()
#
#     def graph_construct(arrays):
#         la = len(arrays)
#         dtype = np.find_common_type([a.dtype for a in arrays], [])
#         arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
#         for i, a in enumerate(np.ix_(*arrays)):
#             arr[..., i] = a
#         return arr.sum(axis=-1)
#
#     graph_sequence = np.array(
#         [graph_construct([pickup_by_hour.values[_], dropoff_by_hour.values[_]]) for _ in
#          range(max(len(pickup_by_hour), len(dropoff_by_hour)))])
#     np.save(graph_series_path, graph_sequence)

def graph_sequence_construct(_data, _node_list, graph_path, time_frame):
    grouped_data_0 = _data.groupby(['pickup_cluster_no', 'dropoff_cluster_no',
                                    (_data['pickup_datetime'].dt.hour // time_frame)]).size()
    grouped_data_1 = _data.groupby(['pickup_cluster_no', 'dropoff_cluster_no',
                                    (_data['dropoff_datetime'].dt.hour // time_frame)]).size()
    grouped_data_2 = _data[_data['pickup_datetime'] == _data['dropoff_datetime']].groupby(
        ['pickup_cluster_no', 'dropoff_cluster_no', (_data['pickup_datetime'].dt.hour // time_frame)]).size()

    graphs = list()
    for gidx in range(24 // time_frame):
        graph = np.ndarray(shape=(len(_node_list), len(_node_list)), dtype=np.float32)
        for pidx, pnode in enumerate(_node_list):
            for didx, dnode in enumerate(_node_list):
                p_count = grouped_data_0.get((pnode, dnode, gidx), 0)
                d_count = grouped_data_1.get((pnode, dnode, gidx), 0)
                union = grouped_data_2.get((pnode, dnode, gidx), 0)
                graph[pidx, didx] = p_count + d_count - union
        graph = preprocessing.normalize(graph, norm='l1') + np.diag([1.] * len(_node_list))
        graphs.append(graph)
    np.save(graph_path, np.array(graphs))


if __name__ == '__main__':
    # taxi_data = preprocess_taxi_data()
    data_path = 'data/taxi-data/all-taxi-data.h5'
    cluster_path = 'data/taxi-data/all-cluster-result.h5'

    to_pickup = 'data/taxi-data/all-pickup-data.h5'
    to_dropoff = 'data/taxi-data/all-dropoff-data.h5'
    to_graph = 'data/taxi-data/all-graph-24.npy'

    # data: pd.DataFrame = pd.read_hdf(data_path)
    data: pd.DataFrame = preprocess_taxi_data(data_path, cluster_result=cluster_path)

    node_list = np.array(
        list(set.union(set(data['pickup_cluster_no'].unique()), set(data['dropoff_cluster_no'].unique()))))
    node_list.sort()

    signal_sequence_construct(data, node_list, key='bike', pickup_path=to_pickup, dropoff_path=to_dropoff, freq='1H')
    graph_sequence_construct(data, node_list, to_graph, time_frame=1)
