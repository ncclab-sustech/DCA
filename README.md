
# DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases

This repository contains the implementation of **DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases**. The method integrates pretraining and spatial graph-based constraints to generate anatomically and functionally meaningful brain atlases.



##  Dependencies

We recommend using Python `==3.9.21`.

All required packages are listed in `DCA/req_trim.txt`. You can install them with:


## DCA Usage
1. Prepare Data
Place preprocessed 4D fMRI volumes in data/fmri/, we have placed a demo fMRI.

Place your ROI masks in data/mask/. This implementation supports customization for gray matter, white matter, and subcortex-specific atlases. We have placed a demo mask.

Ensure data/sub_test.txt contains the list of subject IDs (one per line), we have placed a demo text.

2. Run DCA

```bash
python main.py
```

This will generate subject-level brain parcellations using the provided pretrained model. Results will be saved to results/demo/.

 Command-line Options
 
You can customize key inference settings via arguments in `main.py`. The main options are:

- `-k`, `--n_clusters`: Number of parcels to generate (default: `100`)
- `-e`, `--epoch`: Maximum training epochs (default: `8`)
- `-v`, `--vali`: Whether to keep the best atlas based homogeneity (default: `True`)

⚠️ Validation requires more computing resources. If `--vali` is set to `False`, we recommend using `--epoch < 10` to avoid overfitting.


4. Notes
The pretrained model (swin_model_epoch_30.pth) is automatically loaded if present.

