from __future__ import print_function, division
import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["GOTO_NUM_THREADS"] = "16"

os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools

import argparse
from utils import large_constrcut,build_spatial_graph_and_weights,sparse_spectral_clustering,match_labels,validate
from model import DCA,MySwinUNETR
import time
import nibabel as nib
import torch.nn as nn
import warnings 
import argparse
from torch.nn.parameter import Parameter

warnings.filterwarnings("ignore")


def train_DCA(swin,origin,background_mask,evaluation_data,data_dir):

    model = DCA(
        swin,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        background_mask=background_mask,
        ).to(device)
    start = time.time()        

    for name, param in model.swinunetr.named_parameters():
        if 'c3d' in name or 'out' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.Adam([
        {'params': [
            p for n, p in model.swinunetr.named_parameters() 
            if p.requires_grad
        ],
        'lr': args.lr},
    ])

    data = torch.from_numpy(origin).float() 
    data = data.unsqueeze(0).to(device)  
    data.requires_grad_(True)  
    
    D, H, W = background_mask.shape
    mask = torch.tensor(background_mask).to(device)
    valid_idx = torch.nonzero(mask == 0)
    index_map = -torch.ones(D, H, W, dtype=torch.long)
    index_map[mask == 0] = torch.arange(valid_idx.shape[0])  
    # 6-neighbour
    # offsets = torch.tensor([
    #     [0, 0, 1], [0, 0, -1],
    #     [0, 1, 0], [0, -1, 0],
    #     [1, 0, 0], [-1, 0, 0],
    # ]).to(device)

    # 26-neighbour
    offsets = torch.tensor(
        [
            [dx, dy, dz]
            for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)
            if not (dx == dy == dz == 0)           # exclude the centre voxel
        ],
        dtype=torch.int8,                   
    ).to(device) 

    edge_list = []
    for i, coord in enumerate(valid_idx):
        for offset in offsets:
            neighbor = coord + offset
            x, y, z = neighbor.tolist()
            if 0 <= x < D and 0 <= y < H and 0 <= z < W:
                j = index_map[x, y, z].item()
                if j >= 0:
                    edge_list.append([i, j])

    edge_index = torch.tensor(edge_list).t().contiguous().to(device)  # [2, E]
    
    
    model.train()
    print('start tunning')
    best_h = 0
    best_s = 0
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        x_bar, tmp_s, z,hidden = model(data,epoch)

        ## valid flatten
        mask_flat = background_mask.flatten()  
        valid_mask = mask_flat == 0             
        valid_mask = torch.tensor(valid_mask).to(device)
        
        hidden_permuted = hidden.permute(0, 2, 3, 4, 1) 
        if not hidden_permuted.is_contiguous():
            hidden_permuted = hidden_permuted.contiguous()
        hidden_flat = hidden_permuted.view(-1, hidden.shape[1])  
        recon_permuted = x_bar.permute(0, 2, 3, 4, 1)
        if not recon_permuted.is_contiguous():
            recon_permuted = recon_permuted.contiguous()
        recon_flat = recon_permuted.view(-1, x_bar.shape[1])

        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0] 
        valid_voxels = torch.index_select(hidden_flat, dim=0, index=valid_indices)
        valid_recon = torch.index_select(recon_flat, dim=0, index=valid_indices)

        edge_weight = build_spatial_graph_and_weights(
            edge_index,  # [96, 96, 96]
            valid_voxels
        )

        labels = sparse_spectral_clustering(edge_index.detach().cpu(), edge_weight.detach().cpu(), (torch.max(edge_index)+1).item(), args.n_clusters)

        if epoch > 0:
            labels = match_labels(prev_labels, labels, args.n_clusters)
        prev_labels = labels.copy()

        kl_loss = model.total_loss(data, x_bar, tmp_s, target=torch.tensor(labels).long().to(device))
        
        loss = kl_loss
        print('epoch:',epoch)
        # print('reconstr_loss',recon)
        print('kl_loss',kl_loss) 
        
        loss.backward()
        optimizer.step()
        label_3d = np.full(background_mask.shape, -1)  # Initialize with -1 to mark background areas
        # y_pred = target_distribution(tmp_s).argmax(1).cpu()
        label_3d[background_mask == 0] = labels 
        if args.vali :
            homogeneity = validate(data_dir, label_3d+1) #input numpy
            if best_h < homogeneity:
                best_h = homogeneity
                bset_s = label_3d
            label_3d = bset_s

    # edge_weight = build_spatial_graph_and_weights(
    #     edge_index, 
    #     valid_voxels
    # )
    # labels = weighted_bfs_connected_clustering(edge_index.detach(), edge_weight.detach(), (torch.max(edge_index)+1).item(), args.n_clusters)

    end = time.time() 
    print('Running time: ', end-start)
    return label_3d
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='DCA training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--n_clusters','-k', default=100, type=int) #38
    parser.add_argument('--n_z', default=256, type=int)
    parser.add_argument('--epoch','-e', default=8, type=int)
    parser.add_argument('--dis_mask',default=None)
    parser.add_argument('--vali','-v',default=True) # For small Memory, choose False and set smaller epoch about 8 to avoid overfit.


    args = parser.parse_args()
    print(args)
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    print("gpu counts:", torch.cuda.device_count())
    torch.cuda.empty_cache()
    device = torch.device("cuda" if args.cuda else "cpu")

    subj_list_file = r'data/sub_test.txt'

    with open(subj_list_file, 'r') as f:
        subj_ids = f.read().splitlines()
    
    # import random
    # subj_ids_copy = subj_ids[:]  #shuffle
    # random.shuffle(subj_ids_copy)
    
    # Loop over each subject ID in the list
    for idx, subj_id in enumerate(subj_ids):
        k_cluster = args.n_clusters
        args.n_clusters = k_cluster
        print(f"/n[ {idx+1}/{len(subj_ids)} ]  Processing subject {subj_id} …")

        result_file = fr'results/demo/hcp_{subj_id}_{k_cluster}.npy'
        if os.path.isfile(result_file):
            print(f"  ->  {result_file} already exists, skip.")
            continue                   

        data_dir = fr'data/fmri/{subj_id}_FWHM3.nii.gz'
        dis_mask_dir = fr'data/mask/101915_tissue_mask.nii.gz'  

        dis_mask = nib.load(dis_mask_dir).get_fdata()
        dis_mask = (dis_mask==1) | (dis_mask==11)
        # 1: left grey matter
        # 2: left white matter
        # 3: left subcortex
        # 11: right grey matter
        # 12: right white matter
        # 13: right subcortex
        # 20: corpus callosum
        dis_mask = large_constrcut(dis_mask)  #processing isolate points

        original_volume = nib.load(data_dir).get_fdata().astype('float32')
        original_volume = np.transpose(original_volume, (3, 0, 1, 2))
        evaluation_data = original_volume

        background_mask = (dis_mask == 0)
        background_mask = np.broadcast_to(background_mask, original_volume.shape)
        nb_vals = original_volume[~background_mask]
        if nb_vals.min() != nb_vals.max():
            mean, std = nb_vals.mean(), nb_vals.std()
            original_volume[~background_mask] = (nb_vals - mean) / std
        original_volume[background_mask] = 0
        normalized_original = original_volume

        checkpoint_path = 'data/swin_model_epoch_30.pth'
        model = MySwinUNETR(img_size=(96,96,96), in_channels=300,
                            out_channels=300, feature_size=48,
                            emb_size=256, use_checkpoint=False).to(device)
        if os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  -> checkpoint loaded ({checkpoint_path})")

        label_3d = train_DCA(model,normalized_original,background_mask[0],evaluation_data,data_dir)


        np.save(result_file, label_3d+1)
        print(f"  -> saved {result_file}")
