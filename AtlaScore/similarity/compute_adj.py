import numpy as np
import nibabel as nib
from scipy.ndimage import generate_binary_structure, binary_dilation


def calculate_label_adjacency_bg(atlas_file, exclude_zero=True):
    if atlas_file.endswith('nii.gz'):
        nii = nib.load(atlas_file)
        data = nii.get_fdata().astype(np.int32)
    else:
        data = np.load(atlas_file)
    
    unique_labels = np.unique(data)
    bg_label = np.min(unique_labels)
    print('background value: ', bg_label)

    if exclude_zero:
        unique_labels = unique_labels[unique_labels != bg_label]
    
    adjacency_dict = {label: set() for label in unique_labels}
    
    struct = generate_binary_structure(3, 3)
    
    for label in unique_labels:
        mask = (data == label)
        
        dilated = binary_dilation(mask, structure=struct)
        border = dilated & ~mask  
        
        adjacent_labels = np.unique(data[border])
        if exclude_zero:
            adjacent_labels = adjacent_labels[adjacent_labels != 0]
        
        adjacency_dict[label].update(adjacent_labels)
    
    adjacency_dict = {k: list(v) for k, v in adjacency_dict.items()}
    
    return adjacency_dict



def get_adjacent_voxels(atlas_file, adjacency_dict):
    if atlas_file.endswith('nii.gz'):
        nii = nib.load(atlas_file)
        data = nii.get_fdata().astype(np.int32)
    else:
        data = np.load(atlas_file)
    label_voxels = {}
    for label in adjacency_dict.keys():
        label_voxels[label] = np.array(np.where(data == label)).T
    
    adj_voxel_dict = {}
    
    print(f"total of {len(adjacency_dict)} label adjacency relationship:")
    for label, neighbors in adjacency_dict.items():
        print(f"Processing label {label},  {len(neighbors)} neighbors")
        
        adjacent_voxels = []
        for n in neighbors:
            if n in label_voxels:
                adjacent_voxels.extend(label_voxels[n])

        adj_voxel_dict[label] = np.array(adjacent_voxels) if adjacent_voxels else np.array([])
        
        print(f"Label {label} neighbor: {len(adj_voxel_dict[label])}")
    
    return adj_voxel_dict
