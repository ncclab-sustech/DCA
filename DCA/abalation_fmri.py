import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from utils import process_nifti_file,large_constrcut,build_spatial_graph_and_weights,sparse_spectral_clustering

# ------------------------ Main ------------------------
if __name__=='__main__':
    data_dir = r'./data/fmri'
    mask_dir = r'./data/mask'
    # out_graph_dir = [todo: your out path]
    # out_km_dir    = [todo: your out path]
    os.makedirs(out_graph_dir, exist_ok=True)
    os.makedirs(out_km_dir, exist_ok=True)

    with open(r'./data/sub_test.txt','r') as f:
        subs = [s.strip() for s in f if s.strip()]

    for subj in subs:
        print('Subject', subj)
        mask = nib.load(f'{mask_dir}/{subj}.nii.gz').get_fdata().astype(int)
        dis_mask = (mask==1)|(mask==11)
        dis_mask = large_constrcut(dis_mask)
        vol = nib.load(f'{data_dir}/{subj}_FWHM3.nii.gz').get_fdata().astype(np.float32)
        vol = np.transpose(vol,(3,0,1,2))  # (T,X,Y,Z)
        T,D,H,W = vol.shape[0], *vol.shape[1:]
        ts = vol.reshape(T, -1).T  # (N, T)
        valid_idx = np.where(dis_mask.flatten())[0]
        feats = ts[valid_idx]

        import itertools
        mask = torch.tensor(dis_mask)
        valid_idx = torch.nonzero(mask == 1)
        index_map = -torch.ones(D, H, W, dtype=torch.long)
        index_map[mask == 1] = torch.arange(valid_idx.shape[0])  

        offsets = torch.tensor(
            [
                [dx, dy, dz]
                for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)
                if not (dx == dy == dz == 0)         
            ],
            dtype=torch.int8,                         
        )

        edge_list = []
        for i, coord in enumerate(valid_idx):
            for offset in offsets:
                neighbor = coord + offset
                x, y, z = neighbor.tolist()
                if 0 <= x < D and 0 <= y < H and 0 <= z < W:
                    j = index_map[x, y, z].item()
                    if j >= 0:
                        edge_list.append([i, j])

        ei = torch.tensor(edge_list).t().contiguous() # [2, E]

        ew = build_spatial_graph_and_weights(ei, torch.tensor(feats)) 
        
        valid = np.nonzero(dis_mask.flatten())[0]

        labels_graph = sparse_spectral_clustering(torch.tensor(ei), torch.tensor(ew), len(valid), k=100)
        out_vol = np.zeros(D*H*W, dtype=int)
        out_vol[valid] = labels_graph+1
        out3 = out_vol.reshape(D,H,W)
        np.save(f'{out_graph_dir}/{subj}_graph.npy', out3)

        # Direct KMeans clustering
        km = KMeans(n_clusters=100, n_init=10)
        labels_km = km.fit_predict(feats)
        out_vol_km = np.zeros(D*H*W, dtype=int)
        out_vol_km[valid] = labels_km+1
        out3_km = out_vol_km.reshape(D,H,W)
        np.save(f'{out_km_dir}/{subj}_kmeans.npy', out3_km)

