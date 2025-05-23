import os
from boto3.session import Session
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn.datasets import fetch_abide_pcp
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import zscore
import shutil
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



def get_sub_HCP_rfMRI(subject, access_key, secret_key, addr = './nii_data'):

    bucketName = 'hcp-openaccess'
    prefix = 'HCP_1200/'
    session = Session(aws_access_key_id = access_key, aws_secret_access_key = secret_key)
    bucket = session.resource('s3').Bucket(bucketName)
    os.makedirs(os.path.join(addr, 'HCP_rfMRI'), exist_ok = True)

    for run in ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']:

        source_addr = '{}{}/MNINonLinear/Results/rfMRI_{}/rfMRI_{}_hp2000_clean.nii.gz'.format(prefix, subject, run, run)
        target_addr = os.path.join(addr, 'HCP_rfMRI/{}_rfMRI_{}_hp2000_clean.nii.gz'.format(subject, run))

        if not os.path.exists(target_addr):
            try: bucket.download_file(source_addr, target_addr)
            except: continue



def get_sub_HCP_tfMRI(subject, access_key, secret_key, addr = './nii_data'):

    bucketName = 'hcp-openaccess'
    prefix = 'HCP_1200/'
    session = Session(aws_access_key_id = access_key, aws_secret_access_key = secret_key)
    bucket = session.resource('s3').Bucket(bucketName)
    os.makedirs(os.path.join(addr, 'HCP_tfMRI'), exist_ok = True)

    for task in ['WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']:

        if task == 'WM': subtask_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools']
        elif task == 'GAMBLING': subtask_list = ['win', 'loss']
        elif task == 'MOTOR': subtask_list = ['cue', 'lf', 'rf', 'lh', 'rh', 't']
        elif task == 'LANGUAGE': subtask_list = ['story', 'math']
        elif task == 'SOCIAL': subtask_list = ['mental', 'rnd']
        elif task == 'RELATIONAL': subtask_list = ['relation', 'match']
        else: subtask_list = ['fear', 'neut']

        source_addr = '{}{}/MNINonLinear/Results/tfMRI_{}_LR/tfMRI_{}_LR.nii.gz'.format(prefix, subject, task, task)
        target_addr = os.path.join(addr, 'HCP_tfMRI/{}_tfMRI_{}_LR.nii.gz'.format(subject, task))

        if not os.path.exists(target_addr):
            try: bucket.download_file(source_addr, target_addr)
            except: continue
        
        for subtask in subtask_list:

            subtask_source_addr = '{}{}/MNINonLinear/Results/tfMRI_{}_LR/EVs/{}.txt'.format(prefix, subject, task, subtask)
            subtask_target_addr = os.path.join(addr, 'HCP_tfMRI/{}_tfMRI_{}_{}_ev.txt'.format(subject, task, subtask))

            if not os.path.exists(subtask_target_addr):
                try: bucket.download_file(subtask_source_addr, subtask_target_addr)
                except: continue



def get_ABIDE(addr = './nii_data'):

    fetch_abide_pcp(data_dir = addr, band_pass_filtering = True)
    os.remove(os.path.join(addr, 'README.md'))
    os.remove(os.path.join(addr, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'))
    shutil.move(os.path.join(addr, 'ABIDE_pcp/cpac/filt_noglobal'), os.path.join(addr, 'ABIDE'))
    shutil.rmtree(os.path.join(addr, 'ABIDE_pcp'))



def get_fc_HCP_rfMRI(subject, atlas_name, atlas_loc, nii_addr = './nii_data', fc_addr = './fc_data'):

    os.makedirs(os.path.join(fc_addr, 'HCP_rfMRI', atlas_name), exist_ok = True)

    for run in ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']:
    
        rfmri_img = nib.load(os.path.join(nii_addr, 'HCP_rfMRI', '{}_rfMRI_{}_hp2000_clean.nii.gz'.format(subject, run)))
        rfmri_data = rfmri_img.get_fdata(); X = rfmri_data.reshape(-1, rfmri_data.shape[-1])
        atlas = nib.load(atlas_loc); atlas = resample_to_img(atlas, rfmri_img, interpolation = 'nearest')
        labels = atlas.get_fdata().astype(int).flatten()
        time_series = np.stack([X[labels == label, :].mean(axis = 0) for label in np.sort(np.unique(labels[labels > 0]))], axis = 1)
        time_series = zscore(detrend(time_series, axis = 0, type = 'linear'), axis = 0, ddof = 1)

        for i in range(int(X.shape[1]/300)):

            fc = np.corrcoef(time_series[i*300:(i+1)*300].T); n = fc.shape[0]; fc = fc[np.triu_indices(n, k = 1)]
            if not np.isnan(fc).any(): np.savez_compressed(os.path.join(fc_addr, 'HCP_rfMRI', '{}/{}_{}_{}.npz'.format(atlas_name, subject, run, i)), fc)



def get_fc_HCP_tfMRI(subject, atlas_name, atlas_loc, nii_addr = './nii_data', fc_addr = './fc_data'):

    os.makedirs(os.path.join(fc_addr, 'HCP_tfMRI', atlas_name), exist_ok = True)

    for task in ['WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']:

        if task == 'WM': subtask_list = ['0bk_body', '0bk_faces', '0bk_places', '0bk_tools', '2bk_body', '2bk_faces', '2bk_places', '2bk_tools']
        elif task == 'GAMBLING': subtask_list = ['win', 'loss']
        elif task == 'MOTOR': subtask_list = ['cue', 'lf', 'rf', 'lh', 'rh', 't']
        elif task == 'LANGUAGE': subtask_list = ['story', 'math']
        elif task == 'SOCIAL': subtask_list = ['mental', 'rnd']
        elif task == 'RELATIONAL': subtask_list = ['relation', 'match']
        else: subtask_list = ['fear', 'neut']

        tfmri_img = nib.load(os.path.join(nii_addr, 'HCP_tfMRI', '{}_tfMRI_{}_LR.nii.gz'.format(subject, task)))
        tfmri_data = tfmri_img.get_fdata(); X = tfmri_data.reshape(-1, tfmri_data.shape[-1])
        atlas = nib.load(atlas_loc); atlas = resample_to_img(atlas, tfmri_img, interpolation = 'nearest')
        labels = atlas.get_fdata().astype(int).flatten()
        time_series = np.stack([X[labels == label, :].mean(axis = 0) for label in np.sort(np.unique(labels[labels > 0]))], axis = 1)
        time_series = zscore(detrend(time_series, axis = 0, type = 'linear'), axis = 0, ddof = 1)

        fc = np.corrcoef(time_series.T); n = fc.shape[0]; fc = fc[np.triu_indices(n, k = 1)]
        if not np.isnan(fc).any(): np.savez_compressed(os.path.join(fc_addr, 'HCP_tfMRI', '{}/{}_{}_LR.npz'.format(atlas_name, subject, task)), fc)

        timepoint = np.arange(X.shape[-1]) * 0.72

        for subtask in subtask_list:

            if not os.path.exists(os.path.join(nii_addr, 'HCP_tfMRI', '{}_tfMRI_{}_{}_ev.txt'.format(subject, task, subtask))): continue

            task_timepoint = np.loadtxt(os.path.join(nii_addr, 'HCP_tfMRI', '{}_tfMRI_{}_{}_ev.txt'.format(subject, task, subtask))).reshape(-1, 3)
            subtask_time_series = np.vstack([time_series[(timepoint > task_timepoint[i, 0] - 1e-3) & (timepoint < task_timepoint[i, 0] + task_timepoint[i, 1] + 1e-3)] for i in range(task_timepoint.shape[0])])
            fc = np.corrcoef(subtask_time_series.T); n = fc.shape[0]; fc = fc[np.triu_indices(n, k = 1)]
            if not np.isnan(fc).any(): np.savez_compressed(os.path.join(fc_addr, 'HCP_tfMRI', '{}/{}_{}_LR_{}.npz'.format(atlas_name, subject, task, subtask)), fc)



def get_fc_ABIDE(atlas_name, atlas_loc, nii_addr = './nii_data', fc_addr = './fc_data'):

    os.makedirs(os.path.join(fc_addr, 'ABIDE', atlas_name), exist_ok = True)

    for filename in os.listdir(os.path.join(nii_addr, 'ABIDE')):

        prefix = filename[:-20]

        rfmri_img = nib.load(os.path.join(nii_addr, 'ABIDE', filename))
        rfmri_data = rfmri_img.get_fdata(); X = rfmri_data.reshape(-1, rfmri_data.shape[-1])
        atlas = nib.load(atlas_loc); atlas = resample_to_img(atlas, rfmri_img, interpolation = 'nearest')
        labels = atlas.get_fdata().astype(int).flatten()
        time_series = np.stack([X[labels == label, :].mean(axis = 0) for label in np.sort(np.unique(labels[labels > 0]))], axis = 1)
        time_series = zscore(detrend(time_series, axis = 0, type = 'linear'), axis = 0, ddof = 1)
        fc = np.corrcoef(time_series.T); n = fc.shape[0]; fc = fc[np.triu_indices(n, k = 1)]
        if not np.isnan(fc).any(): np.savez_compressed(os.path.join(fc_addr, 'ABIDE', '{}/{}.npz'.format(atlas_name, prefix)), fc)



def get_fc_ADNI(atlas_name, atlas_loc, nii_addr = './nii_data', fc_addr = './fc_data'):

    os.makedirs(os.path.join(fc_addr, 'ADNI', atlas_name), exist_ok = True)

    for prefix in os.listdir(os.path.join(nii_addr, 'ADNI')):

        rfmri_img = nib.load(os.path.join(nii_addr, 'ADNI', prefix, 'Filtered_4DVolume.nii'))
        rfmri_data = rfmri_img.get_fdata(); X = rfmri_data.reshape(-1, rfmri_data.shape[-1])
        atlas = nib.load(atlas_loc); atlas = resample_to_img(atlas, rfmri_img, interpolation = 'nearest')
        labels = atlas.get_fdata().astype(int).flatten()
        time_series = np.stack([X[labels == label, :].mean(axis = 0) for label in np.sort(np.unique(labels[labels > 0]))], axis = 1)
        time_series = zscore(detrend(time_series, axis = 0, type = 'linear'), axis = 0, ddof = 1)
        fc = np.corrcoef(time_series.T); n = fc.shape[0]; fc = fc[np.triu_indices(n, k = 1)]
        if not np.isnan(fc).any(): np.savez_compressed(os.path.join(fc_addr, 'ADNI', '{}/{}.npz'.format(atlas_name, prefix)), fc)



def fc_stability(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    addr = os.path.join(fc_addr, 'HCP_rfMRI', atlas_name)

    res_list = []
    for sub in subjlist:
        if len([file for file in os.listdir(addr) if sub in file]) > 1:
            fc_corr = np.corrcoef(np.array([np.load(os.path.join(addr, file))['arr_0'] for file in os.listdir(addr) if sub in file]))
            res_list.append(np.mean(fc_corr[np.triu_indices(fc_corr.shape[0], k = 1)]))
    
    return np.array(res_list)



def fingerprinting(atlas_name, fc_addr = './fc_data'):

    addr = os.path.join(fc_addr, 'HCP_rfMRI', atlas_name)
    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    subjlist = [sub for sub in subjlist if [file for file in os.listdir(addr) if sub in file] != []]

    reference = [np.load(os.path.join(addr, [file for file in os.listdir(addr) if sub in file][0]))['arr_0'] for sub in subjlist]
    res_list = []

    for sub in subjlist:
        if len([file for file in os.listdir(addr) if sub in file]) == 1: continue
        file_list = [file for file in os.listdir(addr) if sub in file][1:]
        fc_list = [np.load(os.path.join(addr, file))['arr_0'] for file in file_list]
        count = 0
        for idx in range(len(file_list)):
            temp = np.array([np.corrcoef([fc_list[idx], ref])[0, 1] for ref in reference])
            if subjlist[np.where(temp == np.max(temp))[0][0]] == sub: count += 1
        res_list.append(count/len(fc_list))

    return np.array(res_list)



def age_group_classification(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    behavior = pd.read_csv('./docs/behavior_HCP.csv').to_dict(orient = 'list')

    addr = os.path.join(fc_addr, 'HCP_rfMRI', atlas_name)

    sub_X = []; sub_y = []; mat_X = []; mat_y = []; mat_master = []
    for sub in subjlist:
        target = behavior['Age_Group'][behavior['Subject'].index(int(sub))]
        if np.isnan(target): continue
        sub_X.append(int(sub)); sub_y.append(target)
        file_list = [np.load(os.path.join(addr, file))['arr_0'] for file in os.listdir(addr) if sub in file]
        for f in file_list: mat_X.append(f); mat_y.append(target); mat_master.append(int(sub))
    sub_X = np.array(sub_X); sub_y = np.array(sub_y); mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master)

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainsub_idx, testsub_idx in skf.split(sub_X, sub_y):

        train_sub, test_sub = sub_X[trainsub_idx], sub_X[testsub_idx]
        trainmat_idx = np.array([(train_sub == master).any() for master in mat_master]); testmat_idx = np.array([(test_sub == master).any() for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def gender_classification(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    behavior = pd.read_csv('./docs/behavior_HCP.csv').to_dict(orient = 'list')

    addr = os.path.join(fc_addr, 'HCP_rfMRI', atlas_name)

    sub_X = []; sub_y = []; mat_X = []; mat_y = []; mat_master = []
    for sub in subjlist:
        target = int(behavior['Gender'][behavior['Subject'].index(int(sub))] == 'M')
        if np.isnan(target): continue
        sub_X.append(int(sub)); sub_y.append(target)
        file_list = [np.load(os.path.join(os.path.join(addr, file)))['arr_0'] for file in os.listdir(addr) if sub in file]
        for f in file_list: mat_X.append(f); mat_y.append(target); mat_master.append(int(sub))
    sub_X = np.array(sub_X); sub_y = np.array(sub_y); mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master)

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainsub_idx, testsub_idx in skf.split(sub_X, sub_y):

        train_sub, test_sub = sub_X[trainsub_idx], sub_X[testsub_idx]
        trainmat_idx = np.array([(train_sub == master).any() for master in mat_master]); testmat_idx = np.array([(test_sub == master).any() for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def fluid_intelligence(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    behavior = pd.read_csv('./docs/behavior_HCP.csv').to_dict(orient = 'list')

    addr = os.path.join(fc_addr, 'HCP_rfMRI', atlas_name)

    sub_X = []; sub_y = []; mat_X = []; mat_y = []; mat_master = []
    for sub in subjlist:
        target = behavior['CogFluidComp_AgeAdj_Group'][behavior['Subject'].index(int(sub))]
        if np.isnan(target): continue
        sub_X.append(int(sub)); sub_y.append(target)
        file_list = [np.load(os.path.join(addr, file))['arr_0'] for file in os.listdir(addr) if sub in file]
        for f in file_list: mat_X.append(f); mat_y.append(target); mat_master.append(int(sub))
    sub_X = np.array(sub_X); sub_y = np.array(sub_y); mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master)

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainsub_idx, testsub_idx in skf.split(sub_X, sub_y):

        train_sub, test_sub = sub_X[trainsub_idx], sub_X[testsub_idx]
        trainmat_idx = np.array([(train_sub == master).any() for master in mat_master]); testmat_idx = np.array([(test_sub == master).any() for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def crystallized_intelligence(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    behavior = pd.read_csv('./docs/behavior_HCP.csv').to_dict(orient = 'list')

    addr = os.path.join(fc_addr, 'HCP_rfMRI', atlas_name)

    sub_X = []; sub_y = []; mat_X = []; mat_y = []; mat_master = []
    for sub in subjlist:
        target = behavior['CogCrystalComp_AgeAdj_Group'][behavior['Subject'].index(int(sub))]
        if np.isnan(target): continue
        sub_X.append(int(sub)); sub_y.append(target)
        file_list = [np.load(os.path.join(addr, file))['arr_0'] for file in os.listdir(addr) if sub in file]
        for f in file_list: mat_X.append(f); mat_y.append(target); mat_master.append(int(sub))
    sub_X = np.array(sub_X); sub_y = np.array(sub_y); mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master)

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainsub_idx, testsub_idx in skf.split(sub_X, sub_y):

        train_sub, test_sub = sub_X[trainsub_idx], sub_X[testsub_idx]
        trainmat_idx = np.array([(train_sub == master).any() for master in mat_master]); testmat_idx = np.array([(test_sub == master).any() for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def general_intelligence(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    behavior = pd.read_csv('./docs/behavior_HCP.csv').to_dict(orient = 'list')

    addr = os.path.join(fc_addr, 'HCP_rfMRI', atlas_name)

    sub_X = []; sub_y = []; mat_X = []; mat_y = []; mat_master = []
    for sub in subjlist:
        target = behavior['CogTotalComp_AgeAdj_Group'][behavior['Subject'].index(int(sub))]
        if np.isnan(target): continue
        sub_X.append(int(sub)); sub_y.append(target)
        file_list = [np.load(os.path.join(addr, file))['arr_0'] for file in os.listdir(addr) if sub in file]
        for f in file_list: mat_X.append(f); mat_y.append(target); mat_master.append(int(sub))
    sub_X = np.array(sub_X); sub_y = np.array(sub_y); mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master)

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainsub_idx, testsub_idx in skf.split(sub_X, sub_y):

        train_sub, test_sub = sub_X[trainsub_idx], sub_X[testsub_idx]
        trainmat_idx = np.array([(train_sub == master).any() for master in mat_master]); testmat_idx = np.array([(test_sub == master).any() for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def cognitive_task_7way(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    addr = os.path.join(fc_addr, 'HCP_tfMRI', atlas_name)
    task = ['WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']

    mat_X = []; mat_y = []; mat_master = []
    for sub in subjlist:
        for t in task:
            if not os.path.exists(os.path.join(addr, '{}_{}_LR.npz'.format(sub, t))): continue
            mat_X.append(np.load(os.path.join(addr, '{}_{}_LR.npz'.format(sub, t)))['arr_0']); mat_y.append(task.index(t)); mat_master.append(sub)
    mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master)

    kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainsub_idx, testsub_idx in kf.split(subjlist):

        train_sub, test_sub = subjlist[trainsub_idx], subjlist[testsub_idx]
        trainmat_idx = np.array([(train_sub == master).any() for master in mat_master]); testmat_idx = np.array([(test_sub == master).any() for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def cognitive_task_24way(atlas_name, fc_addr = './fc_data'):

    subjlist = np.loadtxt('./docs/HCP_subjlist.txt', dtype = str)
    addr = os.path.join(fc_addr, 'HCP_tfMRI', atlas_name)
    task = {
        '0bk_body': 'WM', '0bk_faces': 'WM', '0bk_places': 'WM', '0bk_tools': 'WM', '2bk_body': 'WM', '2bk_faces': 'WM', '2bk_places': 'WM', '2bk_tools': 'WM', 
        'win': 'GAMBLING', 'loss': 'GAMBLING', 
        'cue': 'MOTOR', 'lf': 'MOTOR', 'rf': 'MOTOR', 'lh': 'MOTOR', 'rh': 'MOTOR', 't': 'MOTOR', 
        'story': 'LANGUAGE', 'math': 'LANGUAGE', 
        'mental': 'SOCIAL', 'rnd': 'SOCIAL', 
        'relation': 'RELATIONAL', 'match': 'RELATIONAL', 
        'fear': 'EMOTION', 'neut': 'EMOTION'
    }

    mat_X = []; mat_y = []; mat_master = []
    for sub in subjlist:
        for t in list(task.keys()):
            if not os.path.exists(os.path.join(addr, '{}_{}_LR_{}.npz'.format(sub, task[t], t))): continue
            mat_X.append(np.load(os.path.join(addr, '{}_{}_LR_{}.npz'.format(sub, task[t], t)))['arr_0']); mat_y.append(list(task.keys()).index(t)); mat_master.append(sub)
    mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master)

    kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainsub_idx, testsub_idx in kf.split(subjlist):

        train_sub, test_sub = subjlist[trainsub_idx], subjlist[testsub_idx]
        trainmat_idx = np.array([(train_sub == master).any() for master in mat_master]); testmat_idx = np.array([(test_sub == master).any() for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def autism_diagnosis(atlas_name, fc_addr = './fc_data'):

    behavior = pd.read_csv('./docs/behavior_ABIDE.csv').to_dict(orient = 'list')
    addr = os.path.join(fc_addr, 'ABIDE', atlas_name)

    mat_X = []; mat_y = []
    for file in os.listdir(addr):
        target = behavior['DX_GROUP'][behavior['FILE_ID'].index(file[:-4])]
        if np.isnan(target): continue
        mat_X.append(np.load(os.path.join(addr, file))['arr_0']); mat_y.append(target)
    mat_X = np.array(mat_X); mat_y = np.array(mat_y)

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainmat_idx, testmat_idx in skf.split(mat_X, mat_y):

        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def autism_cross_site(atlas_name, fc_addr = './fc_data'):

    behavior = pd.read_csv('./docs/behavior_ABIDE.csv').to_dict(orient = 'list')
    addr = os.path.join(fc_addr, 'ABIDE', atlas_name)

    mat_X = []; mat_y = []; mat_master = []
    for file in os.listdir(addr):
        target = behavior['DX_GROUP'][behavior['FILE_ID'].index(file[:-4])]
        site = behavior['SITE_ID'][behavior['FILE_ID'].index(file[:-4])]
        if np.isnan(target): continue
        mat_X.append(np.load(os.path.join(addr, file))['arr_0']); mat_y.append(target); mat_master.append(site)
    mat_X = np.array(mat_X); mat_y = np.array(mat_y); mat_master = np.array(mat_master, dtype = str)

    acc_list = []

    for holdout_site in np.unique(mat_master):

        trainmat_idx = np.array([(master != holdout_site) for master in mat_master]); testmat_idx = np.array([(master == holdout_site) for master in mat_master])
        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def AD_diagnosis(atlas_name, fc_addr = './fc_data'):

    behavior = pd.read_csv('./docs/behavior_ADNI.csv').to_dict(orient = 'list')
    addr = os.path.join(fc_addr, 'ADNI', atlas_name)

    mat_X = []; mat_y = []
    for file in os.listdir(addr):
        target = behavior['Research Group'][behavior['Subject ID'].index(file[:-4])]
        if target == 'CN': target_idx = 0
        elif target == 'MCI': target_idx = 1
        else: target_idx = 2
        mat_X.append(np.load(os.path.join(addr, file))['arr_0']); mat_y.append(target_idx)
    mat_X = np.array(mat_X); mat_y = np.array(mat_y)

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
    acc_list = []

    for trainmat_idx, testmat_idx in skf.split(mat_X, mat_y):

        X_train, X_test = mat_X[trainmat_idx], mat_X[testmat_idx]; y_train, y_test = mat_y[trainmat_idx], mat_y[testmat_idx]

        if X_train.shape[1] > 100:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            pca = PCA(n_components = 100, random_state = 0); X_train_pca = pca.fit_transform(X_train_scaled); X_test_pca = pca.transform(X_test_scaled)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_pca, y_train)

            y_pred = clf.predict(X_test_pca)
            acc_list.append(accuracy_score(y_test, y_pred))

        else:

            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

            clf = SVC(kernel = 'linear', class_weight = 'balanced')
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_test_scaled)
            acc_list.append(accuracy_score(y_test, y_pred))

    return np.array(acc_list)



def downstream_all(atlas_name, fc_addr = './fc_data'):

    print('--- {} downstream report ---'.format(atlas_name))

    res = gender_classification(atlas_name, fc_addr)
    print('Gender classification: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = fluid_intelligence(atlas_name, fc_addr)
    print('Fluid intelligence: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = cognitive_task_7way(atlas_name, fc_addr)
    print('Cognitive task (7-way): {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = cognitive_task_24way(atlas_name, fc_addr)
    print('Cognitive task (24-way): {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = autism_diagnosis(atlas_name, fc_addr)
    print('Autism diagnosis: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = AD_diagnosis(atlas_name, fc_addr)
    print('AD diagnosis: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = fc_stability(atlas_name, fc_addr)
    print('FC stability: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = fingerprinting(atlas_name, fc_addr)
    print('Fingerprinting: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = age_group_classification(atlas_name, fc_addr)
    print('Age group classification: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = crystallized_intelligence(atlas_name, fc_addr)
    print('Crystallized intelligence: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = general_intelligence(atlas_name, fc_addr)
    print('General intelligence: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))

    res = autism_cross_site(atlas_name, fc_addr)
    print('Autism cross-site: {:.3f}±{:.3f}'.format(np.mean(res), np.std(res)))