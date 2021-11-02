import os
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from sklearn.decomposition import LatentDirichletAllocation
num_ontology_size = 5861
num_lda_size = 100


def compute_data_split(n, random=True, **kwargs):
    if random:
        raise NotImplementedError
    elif 'split' in kwargs:
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        split = kwargs['split']
        train_mask[split[0]] = True
        val_mask[split[1]] = True
        test_mask[split[2]] = True
    else:
        raise NotImplementedError

    return train_mask, val_mask, test_mask


def batch_data_list(data_list, **kwargs):
    graph_list = []
    for i in range(len(data_list)):
        try:
            ontology_feature = data_list[i][6]
        except:
            ontology_feature = np.zeros(num_ontology_size)
        try:
            lda_feature = data_list[i][7]
        except:
            lda_feature = np.zeros(num_lda_size)
        graph_list.append(Data(
            x=torch.from_numpy(data_list[i][2][1]).float(),
            edge_index=torch.from_numpy(data_list[i][2][0]).long(),
            edge_attr=torch.from_numpy(np.argmax(data_list[i][2][2], axis=-1)).long(),
            y=torch.from_numpy(data_list[i][4]).float().unsqueeze(0),
            instrument=torch.from_numpy(data_list[i][3]).float().unsqueeze(0),
            fp=torch.from_numpy(data_list[i][5]).float().unsqueeze(0),
            shift=torch.from_numpy(np.array([data_list[i][1]])).long(),
            lda_feature=torch.from_numpy(lda_feature).float().unsqueeze(0),
            ontology_feature=torch.from_numpy(ontology_feature).float().unsqueeze(0),
        )
        )

    return Batch.from_data_list(graph_list)


def convert_candidate_to_data_list(source_data, candidate_list):
    n = len(candidate_list)

    for i in range(n):
        candidate_list[i] = [
            '?',
            source_data[1],
            candidate_list[i][1],
            source_data[3],
            source_data[4],
            candidate_list[i][2]
        ]

    return candidate_list


def compute_lda_feature(data, lda_component=100, saved_file_path='./data/lda'):
    saved_file_path_full = "_".join([saved_file_path, str(lda_component)])

    if os.path.exists(saved_file_path_full + '.npz'):
        data = np.load(saved_file_path_full + '.npz')
        lda_topic = data['lda_topic']
        return lda_topic

    data = data / np.maximum(1e-6, data.max(axis=1, keepdims=True)) * 100.0
    data = data.astype(int)

    lda = LatentDirichletAllocation(n_components=lda_component)
    lda.fit(data)
    lda_topic = lda.transform(data)

    np.savez(saved_file_path_full, lda_topic=lda_topic)

    return lda_topic
