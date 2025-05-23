import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
#   pip install monai
from monai.networks.nets import SwinUNETR
import importlib.util
spec = importlib.util.spec_from_file_location('SwinUNETR', 'swin_unetr.py')
my_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(my_module)
SwinUNETR = my_module.SwinUNETR

class MySwinUNETR(SwinUNETR):
    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        out = self.c3d(out)
        logits = self.out(out)
        return logits, out

class DCA(nn.Module):
    def __init__(self,
                 model,
                 n_z,
                 n_clusters,
                 background_mask,
                 ):
        super(DCA, self).__init__()
        # self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.swinunetr = model
        self.background_mask = background_mask
        self.D = Parameter(torch.Tensor(n_clusters, n_z),requires_grad=True)
        nn.init.xavier_uniform_(self.D)

     
    def forward(self, x, epoch):
        x_bar, hidden = self.swinunetr(x)
        z = hidden
        B, C, D, H, W = z.shape
        z_perm = z.permute(0,2,3,4,1).reshape(-1, C)  # (B*D*H*W, C)
        mask_flat = torch.from_numpy(self.background_mask).flatten().to(z.device)  # (D*H*W,)
        z_features = z_perm[~mask_flat.repeat(B)]
        feature_distances = torch.cdist(z_features, self.D)
        s = - feature_distances
        return x_bar, s, z_features,z

    def orthogonality_loss(self,D):
    
        K = D.size(0)
        G = D @ D.t()                     # shape [K, K]
        G_off = G - torch.eye(K, device=D.device, dtype=D.dtype)
        loss = (G_off.sum() / (K * (K - 1))).sqrt() 
        return loss

    def masked_mse_loss(self, pred, target, mask):
        mask = torch.from_numpy(mask)
        if mask.dim() != pred.dim():  # (H, W, D)
            mask = mask.unsqueeze(0)  # Add batch dimension to mask (1, C, H, W, D)
            mask = mask.expand(pred.size(1), -1, -1, -1)  # Repeat mask for each batch (B, C, H, W, D)
            mask = mask.unsqueeze(0)  # Add batch dimension to mask (1, C, H, W, D)
            mask = mask.expand(pred.size(0), -1, -1, -1, -1)
        diff = pred - target
        masked_diff = diff[~mask]  # Select only the non-background areas (where mask is False)
        if masked_diff.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        return torch.mean(masked_diff**2)

    def total_loss(self, x, x_bar, pred, target):

        loss1 = F.cross_entropy(pred, target) #Since target is hard label, minimizing the KL divergence is equivalent to minimizing the cross‐entropy.

        # reconstr_loss =  self.masked_mse_loss(x_bar, x, self.background_mask)    
        # orth_loss = self.orthogonality_loss(self.D)
        return loss1

