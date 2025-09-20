import nibabel as nib
import numpy as np
import pyzstd,io
import pickle
import cupy as cp
from compute_adj import calculate_label_adjacency_bg, get_adjacent_voxels

def load_zstd_file(file_path):
        with open(file_path, 'rb') as f:
            decompressed_data = pyzstd.decompress(f.read())
            buffer = io.BytesIO(decompressed_data)
            return np.load(buffer)
        

def load_dict_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
        

def evaluate(input_file, label_file, adj_dict):

    if input_file.endswith('.zst'):
        input_matrix = load_zstd_file(input_file)
    else:
        input_matrix = nib.load(input_file).get_fdata()
    input_matrix = input_matrix[:,:,:,:300]
    # print(input_matrix.shape) # (96, 96, 96, 60)

    # img_label = nib.load(label_file)
    if label_file.endswith('nii.gz'):
        label_matrix = nib.load(label_file)
        label_matrix = label_matrix.get_fdata()
    else:
        label_matrix = np.load(label_file)
    # print(f'Unique labels in {label_file}:', np.unique(label_matrix))

    unique_labels = np.unique(label_matrix)
    bg = np.min(unique_labels)
    print('background value: ', bg)
    unique_labels = unique_labels[unique_labels > bg]
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

        dissimilarity = 1 - fc_mat
        w_i = np.nanmean(dissimilarity, axis=1)
        adj_voxel_indices = adj_dict[label]
        if adj_voxel_indices.shape[0] == 0:
            silhouette_scores[label] = np.nan
            continue
        adj_time_series = input_matrix[adj_voxel_indices[:, 0], adj_voxel_indices[:, 1], adj_voxel_indices[:, 2], :]

        time_series_gpu = cp.asarray(time_series)
        adj_time_series_gpu = cp.asarray(adj_time_series)
        time_series_std = (time_series_gpu - cp.mean(time_series_gpu, axis=1, keepdims=True)) / cp.std(time_series_gpu, axis=1, keepdims=True)
        adj_time_series_std = (adj_time_series_gpu - cp.mean(adj_time_series_gpu, axis=1, keepdims=True)) / cp.std(adj_time_series_gpu, axis=1, keepdims=True)
        cross_fc_gpu = cp.dot(time_series_std.astype(cp.float32), adj_time_series_std.T.astype(cp.float32)) / time_series_gpu.shape[1]
        cross_fc = cross_fc_gpu.get()
        cp.get_default_memory_pool().free_all_blocks()
        cross_dissimilarity = 1 - cross_fc
        b_i = np.nanmean(cross_dissimilarity, axis=1)
        silhouette = np.zeros(n_voxels)
        valid_mask = (w_i != 0) | (b_i != 0)
        silhouette[valid_mask] = (b_i[valid_mask] - w_i[valid_mask]) / np.maximum(w_i[valid_mask], b_i[valid_mask])
        silhouette_scores[label] = np.nanmean(silhouette)
        
    weighted_sum_fc = 0.0
    valid_voxels = 0 
    for label in fc_means:
        if not np.isnan(fc_means[label]):  
            weighted_sum_fc += fc_means[label] * voxel_counts[label]
            valid_voxels += voxel_counts[label]
    weighted_mean_fc = weighted_sum_fc / valid_voxels if valid_voxels > 0 else np.nan
    
    weighted_sum_silhouette = 0.0
    valid_voxels = 0
    for label in silhouette_scores:
        if not np.isnan(silhouette_scores[label]):
            weighted_sum_silhouette += silhouette_scores[label] * voxel_counts[label]
            valid_voxels += voxel_counts[label]
    silhouette_avg = weighted_sum_silhouette / valid_voxels if valid_voxels > 0 else np.nan

    print(f'Homogeneity: {weighted_mean_fc:.3f}\n')
    print(f'Silhouette Score: {silhouette_avg: .3f}\n')

    return weighted_mean_fc, silhouette_avg

def calculate_fc_mean(time_series):
    if time_series.shape[0] < 2:
        return np.nan, np.nan
    
    time_series_gpu = cp.asarray(time_series)
    fc_matrix = cp.corrcoef(time_series_gpu)
    fc_matrix = fc_matrix.get()
    
    if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
        return np.nan, np.nan 
    
    lower_triangle_indices = np.tril_indices(fc_matrix.shape[0], -1)
    lower_triangle_values = fc_matrix[lower_triangle_indices]
    
    positive_fc_values = lower_triangle_values[lower_triangle_values > 0]
    mean_fc = np.mean(positive_fc_values) if positive_fc_values.size > 0 else 0
    # mean_fc = np.nanmean(lower_triangle_values)
    return mean_fc, fc_matrix

    

if __name__ == "__main__":
    # fmri data
    data_file_path = r'/path/to/your/data'
    # your atlas
    label_file_path = r'/path/to/your/atlas'
    adjacency = calculate_label_adjacency_bg(label_file_path)
    adj_dict = get_adjacent_voxels(label_file_path, adjacency)
    homogeneity, silhouette = evaluate(data_file_path, label_file_path, adj_dict)
