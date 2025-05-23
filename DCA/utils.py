import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from collections import defaultdict
import numpy as np
import torch
from scipy.sparse.linalg import svds


def weighted_bfs_connected_clustering(edge_index, edge_weight, num_nodes, k, seed=None):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    if torch.is_tensor(edge_weight):
        edge_weight = edge_weight.cpu().numpy()

    graph = defaultdict(list)
    E = edge_index.shape[1]
    for idx in range(E):
        i, j = int(edge_index[0, idx]), int(edge_index[1, idx])
        w = float(edge_weight[idx])
        graph[i].append((j, w))
        graph[j].append((i, w))

    labels = np.full(num_nodes, -1, dtype=int)
    max_size = int(np.ceil(num_nodes / k))
    cluster_sizes = [0] * k
    heaps = [ [] for _ in range(k) ]

    seeds = np.random.choice(num_nodes, k, replace=False)
    for cid, seed_node in enumerate(seeds):
        labels[seed_node] = cid
        cluster_sizes[cid] += 1
        for nbr, w in graph[seed_node]:
            if labels[nbr] == -1:
                heapq.heappush(heaps[cid], (-w, nbr))

    assigned = k  

    while assigned < num_nodes:
        made_progress = False
        for cid in range(k):
            if cluster_sizes[cid] >= max_size:
                continue
            heap = heaps[cid]
            while heap:
                neg_w, node = heapq.heappop(heap)
                if labels[node] == -1:
                    labels[node] = cid
                    cluster_sizes[cid] += 1
                    assigned += 1
                    for nbr, w in graph[node]:
                        if labels[nbr] == -1:
                            heapq.heappush(heap, (-w, nbr))
                    made_progress = True
                    break
        if not made_progress:
            break

    for node in range(num_nodes):
        if labels[node] == -1:
            nbr_labels = [labels[nbr] for nbr, _ in graph[node] if labels[nbr] != -1]
            if nbr_labels:
                cid = min(nbr_labels, key=lambda x: cluster_sizes[x])
            else:
                cid = int(np.argmin(cluster_sizes))
            labels[node] = cid
            cluster_sizes[cid] += 1

    return labels


from scipy.optimize import linear_sum_assignment

def match_labels(y_true, y_pred, n_clusters):

    cost = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost[i, j] = np.sum((y_true == i) & (y_pred == j))
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = dict(zip(col_ind, row_ind))
    return np.vectorize(lambda x: mapping.get(x, x))(y_pred)



from scipy import ndimage

def large_constrcut(mask):

    binary_mask = (mask > 0).astype(int)

    structure = ndimage.generate_binary_structure(3, 1)  
    labeled_mask, num_labels = ndimage.label(binary_mask, structure=structure)

    region_sizes = ndimage.sum(binary_mask, labeled_mask, index=np.arange(1, num_labels + 1))

    if num_labels == 0:
        largest_label = 0
    else:
        largest_label = np.argmax(region_sizes) + 1  
    large_mask = (labeled_mask == largest_label).astype(int)     
    print('total voxels:', len(np.where(large_mask>0)[0]))
    return large_mask


def build_spatial_graph_and_weights(edge_index,features):

    src, tgt = edge_index
    feat_src = features[src]
    feat_tgt = features[tgt]
    x_centered = feat_src - feat_src.mean()
    y_centered = feat_tgt - feat_tgt.mean()
    edge_weight = F.cosine_similarity(x_centered, y_centered)
    return edge_weight



import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.cluster import SpectralClustering

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import lobpcg
from sklearn.cluster import AgglomerativeClustering

def sparse_spectral_clustering(edge_index, edge_weight, num_nodes, k=2):

    adj_sparse = to_scipy_sparse_matrix(
        edge_index, edge_weight, num_nodes=num_nodes
    )  

    degree = np.array(adj_sparse.sum(axis=1)).flatten()  
    D = csr_matrix((degree, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes))
    L = D - adj_sparse  

    _, eigenvectors = eigsh(L.astype(np.float64), k=k, which='LM',sigma=0, maxiter=10000) 

    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(eigenvectors)
    return labels



def validate(input_file, label_file):

    if input_file.endswith('.zst'):
        input_matrix = load_zstd_file(input_file)
    elif input_file.endswith('.npy'):
        input_matrix = np.load(input_file)
    else:
        input_matrix = nib.load(input_file).get_fdata()
    input_matrix = input_matrix[:,:,:,:300]

    label_matrix = label_file

    unique_labels = np.unique(label_matrix)
    unique_labels = unique_labels[unique_labels > 0]
    print('roi number:',len(unique_labels))
    fc_means = {}
    voxel_counts = {}
    silhouette_scores = {} 

    for label in unique_labels:
        voxel_indices = np.argwhere(label_matrix == label)
        if voxel_indices.shape[0] < 2:
            fc_means[label] = np.nan
            voxel_counts[label] = len(voxel_indices)
            silhouette_scores[label] = np.nan
            continue
        time_series = input_matrix[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2], :]
        n_voxels = time_series.shape[0]

        fc_mean, fc_mat = calculate_fc_mean(time_series)
        fc_means[label] = fc_mean
        voxel_counts[label] = len(voxel_indices)

    weighted_sum_fc = 0.0
    valid_voxels = 0 
    for label in fc_means:
        if not np.isnan(fc_means[label]):  
            weighted_sum_fc += fc_means[label] * voxel_counts[label]
            valid_voxels += voxel_counts[label]
    weighted_mean_fc = weighted_sum_fc / valid_voxels if valid_voxels > 0 else np.nan


    return weighted_mean_fc


def calculate_fc_mean(time_series):
    if time_series.shape[0] < 2:
        return np.nan, np.nan
    
    fc_matrix = np.corrcoef(time_series)
    
    if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
        return np.nan, np.nan  
    
    lower_triangle_indices = np.tril_indices(fc_matrix.shape[0], -1)
    lower_triangle_values = fc_matrix[lower_triangle_indices]
    
    # positive_fc_values = lower_triangle_values[lower_triangle_values > 0]
    # mean_fc = np.mean(positive_fc_values) if positive_fc_values.size > 0 else 0
    mean_fc = np.mean(lower_triangle_values)
    return mean_fc, fc_matrix
