import scipy
from scipy.sparse import csr_matrix
import torch as pt
import cupy as cp
import numpy as np
import os
import pickle
from pathlib import Path
import nibabel as nib


def compute_var_cov_pt(data, cond='all', mean_centering=True):
    """ Compute the variance and covariance for a given data matrix.
        (PyTorch GPU version)

    Args:
        data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        cond: specify the subset of activation conditions to evaluation
              (e.g condition column [1,2,3,4]), if not given, default to
              use all conditions
        mean_centering: boolean value to determine whether the given subject
                        data should be mean centered

    Returns: cov - the covariance matrix of current subject data, shape [N * N]
             var - the variance matrix of current subject data, shape [N * N]
    """
    if mean_centering:
        data = data - pt.mean(data, dim=1, keepdim=True)  # mean centering
    # specify the condition index used to compute correlation, otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")
    k = data.shape[1]
    cov = pt.matmul(data, data.T) / (k - 1)
    # sd = data.std(dim=1).reshape(-1, 1)  # standard deviation
    sd = pt.sqrt(pt.sum(data ** 2, dim=1, keepdim=True) / (k - 1))
    var = pt.matmul(sd, sd.T)
    return cov, var

def compute_var_cov_np(data, cond='all', mean_centering=True):
    """ Compute the variance and covariance for a given data matrix.
        (Numpy CPU version)

    Args:
        data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        cond: specify the subset of activation conditions to evaluation
              (e.g condition column [1,2,3,4]), if not given, default to
              use all conditions
        mean_centering: boolean value to determine whether the given subject
                        data should be mean centered

    Returns: cov - the covariance matrix of current subject data, shape [N * N]
             var - the variance matrix of current subject data, shape [N * N]
    """
    if mean_centering:
        mean = data.mean(axis=1)
        data = data - mean[:, np.newaxis]  # mean centering
    else:
        data = data

    # specify the condition index used to compute correlation,
    # otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    k = data.shape[1]
    sd = np.sqrt(np.sum(np.square(data), axis=1) / (k-1))  # standard deviation
    sd = np.reshape(sd, (sd.shape[0], 1))
    var = np.matmul(sd, sd.transpose())
    cov = np.matmul(data, data.transpose()) / (k-1)
    return cov, var

def compute_var_cov_cp(data, cond='all', mean_centering=True):
    """ Compute the variance and covariance for a given data matrix.
        (Numpy CPU version)

    Args:
        data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        cond: specify the subset of activation conditions to evaluation
              (e.g condition column [1,2,3,4]), if not given, default to
              use all conditions
        mean_centering: boolean value to determine whether the given subject
                        data should be mean centered

    Returns: cov - the covariance matrix of current subject data, shape [N * N]
             var - the variance matrix of current subject data, shape [N * N]
    """
    if mean_centering:
        mean = data.mean(axis=1)
        data = data - mean[:, cp.newaxis]  # mean centering
    else:
        data = data

    # specify the condition index used to compute correlation,
    # otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")
    data = cp.asarray(data)
    k = data.shape[1]
    sd = cp.sqrt(cp.sum(cp.square(data), axis=1) / (k-1))  # standard deviation
    sd = cp.reshape(sd, (sd.shape[0], 1))
    var = cp.matmul(sd, sd.transpose())
    cov = cp.matmul(data, data.transpose()) / (k-1)
    return cov, var

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def delete_cols_csr(mat, indices):
    """
    Remove the cols denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[1], dtype=bool)
    mask[indices] = False
    return mat[:, mask]


def ecbc(data, dist, parcellation):
        """
        The public function that handle the main DCBC evaluation routine

        :param parcellation: The cortical parcellation to evaluate
        :return: dict T that contain all needed DCBC evaluation results
        """
        maxDist = 35
        binWidth = 1
        weighting = True
        numBins = int(np.floor(maxDist / binWidth))
        h = 'L'

        D = dict()

        dist = csr_matrix(dist)
        dist = dist.tocsr()
        
        print(f'Evaluating {h} hemisphere for subject')

        # remove nan value and medial wall from subject data
        nanIdx = np.union1d(np.unique(np.where(np.isnan(data))[0]), np.where(parcellation == 0)[0])
        data = np.delete(data, nanIdx, axis=0)
        cov, var = compute_var_cov_cp(data)  # This line can be changed to use compute_corr()

        # remove the nan value and medial wall from dist file
        this_dist = delete_rows_csr(dist, nanIdx)
        this_dist = delete_cols_csr(this_dist, nanIdx)
        row, col, distance = scipy.sparse.find(this_dist)

        parcellation = cp.asarray(parcellation)
        row = cp.asarray(row)
        col = cp.asarray(col)
        distance = cp.asarray(distance)

        # making parcellation matrix without medial wall and nan value
        par = cp.delete(parcellation, nanIdx, axis=0)
        num_within, num_between, corr_within, corr_between = [], [], [], []
        for i in range(numBins):
            inBin = cp.where((distance > i * binWidth) & (distance <= (i + 1) * binWidth))[0]

            # lookup the row/col index of within and between vertices
            within = cp.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
            between = cp.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

            # retrieve and append the number of vertices for within/between in current bin
            num_within = cp.append(num_within, within.shape[0])
            num_between = cp.append(num_between, between.shape[0])

            # Compute and append averaged within- and between-parcel correlations in current bin
            this_corr_within = cp.nanmean(cov[row[inBin[within]], col[inBin[within]]]) / cp.nanmean(var[row[inBin[within]], col[inBin[within]]])
            this_corr_between = cp.nanmean(cov[row[inBin[between]], col[inBin[between]]]) / cp.nanmean(var[row[inBin[between]], col[inBin[between]]])
            corr_within = cp.append(corr_within, this_corr_within)
            corr_between = cp.append(corr_between, this_corr_between)

            del inBin

        if weighting:
            weight = 1/(1/num_within + 1/num_between)
            weight = weight / cp.sum(weight)
            DCBC = cp.nansum(cp.multiply((corr_within - corr_between), weight))
        else:
            DCBC = cp.nansum(corr_within - corr_between)
            weight = cp.nan

        D = {
            "binWidth": binWidth,
            "maxDist": maxDist,
            "hemisphere": h,
            "num_within": num_within,
            "num_between": num_between,
            "corr_within": corr_within,
            "corr_between": corr_between,
            "weight": weight,
            "DCBC": DCBC
        }

        return D



if __name__ == "__main__":
    hemisphere = 'L'
    geodesic = np.load('/path/to/your/dist/file') # npz file
    dist = geodesic['arr_0']
    print(f'Geodesic distance shape: {dist.shape}')

    parcellation = nib.load('/path/to/your/atlas/on/surface').darrays[0].data
    print(f'Parcellation shape: {parcellation.shape}')

    data = nib.load('/path/to/your/fmri/on/surface')
    surf = [x.data for x in data.darrays]
    surf = np.array(surf).T
    print(f'FMRI data shape: {surf.shape}')
    
    dcbc = ecbc(surf, dist, parcellation)
    print(dcbc['DCBC'])
