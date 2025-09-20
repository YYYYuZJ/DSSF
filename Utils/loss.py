import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.utils import show_memory

## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum') 
    def forward(self, pred, target, umask, mask_m=None, first_stage=True):
        """
        pred -> [batch*seq_len, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1, 1)
        mask = umask.clone()

        if mask_m == None:
            mask_m = mask
        mask_m = mask_m.reshape(-1, 1)

        target = target.view(-1, 1)
        pred = F.log_softmax(pred, 1)
        loss = self.loss(pred * mask * mask_m, (target * mask * mask_m).squeeze().long()) / torch.sum(mask * mask_m)
        if torch.isnan(loss) == True:
            loss = 0
        return loss

class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum') 

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len, n_calsses(1)]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1, 1) 
        mask = umask.clone()
        
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        
        loss = self.loss(pred*mask, target*mask) / torch.sum(mask)
        if torch.isnan(loss) == True:
            loss = 0
        return loss

class MaskedBaseCLLoss(nn.Module):
    def __init__(self, margin=0.1, dataset='CMUMOSI', device='cpu'):
        super(MaskedBaseCLLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.dataset = dataset
        self.device = device 
    
    def compute_contrastive_loss(self, sim_matrix, N_valid):

        pos_mask = torch.eye(N_valid, dtype=torch.bool, device=self.device)
        pos_sim = sim_matrix[pos_mask].view(N_valid, 1) 
        neg_mask = ~pos_mask
        neg_sim = sim_matrix[neg_mask].view(N_valid, -1) 

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        targets = torch.zeros(N_valid, dtype=torch.long, device=self.device) 

        loss = F.cross_entropy(logits, targets, reduction='mean')
        return loss

    def forward(self, shared_A, shared_T, shared_V, temperature, uumask):
        """
        shared_A -> [batch, seq_len, dim] or [N, dim]
        shared_T -> [batch, seq_len, dim] or [N, dim]
        shared_V -> [batch, seq_len, dim] or [N, dim]
        labels -> [batch, seq_len] or [N]
        uumask -> [batch, seq_len] or [N]
        temperature -> scalar, temperature parameter for scaling similarities
        """
        umask = uumask.reshape(-1, 1)
        shared_A = shared_A.reshape(-1, shared_A.size(-1)) 
        shared_T = shared_T.reshape(-1, shared_T.size(-1)) 
        shared_V = shared_V.reshape(-1, shared_V.size(-1))

        valid_mask = umask.squeeze() > 0  
        shared_A = shared_A[valid_mask] 
        shared_T = shared_T[valid_mask]  
        shared_V = shared_V[valid_mask] 

        shared_A = F.normalize(shared_A, dim=-1)
        shared_T = F.normalize(shared_T, dim=-1)
        shared_V = F.normalize(shared_V, dim=-1)

        sim_AT = torch.matmul(shared_A, shared_T.t()) / temperature
        sim_AV = torch.matmul(shared_A, shared_V.t()) / temperature
        sim_TV = torch.matmul(shared_T, shared_V.t()) / temperature

        N_valid = shared_A.size(0)
        loss_AT = self.compute_contrastive_loss(sim_AT, N_valid)
        loss_AV = self.compute_contrastive_loss(sim_AV, N_valid)
        loss_TV = self.compute_contrastive_loss(sim_TV, N_valid)

        total_loss = (loss_AT + loss_AV + loss_TV) / 3.0

        return total_loss
    
class MaskedSoftCLLoss(torch.nn.Module):
    def __init__(self, margin=0.1, dataset='CMUMOSI', device='cpu', ema_momentum=0.99, warmup_steps=100, start_momentum=0.5):
        super(MaskedSoftCLLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.dataset = dataset
        self.device = device
        self.ema_momentum = ema_momentum
        self.warmup_steps = warmup_steps
        self.start_momentum = start_momentum 
        self.global_centers = {'A': None, 'T': None, 'V': None}
        self.batch_count = 0
        
    def compute_modality_wise_centers(self, shared_A, shared_T, shared_V, labels, preds, umask, top_k=0.5, num_classes=2):
        centers = {'A': None, 'T': None, 'V': None}
        modalities = {'A': shared_A, 'T': shared_T, 'V': shared_V}

        for modality, feat in modalities.items():
            feature_dim = feat.size(-1)
            cls_centers = []
            for cls in range(num_classes):
                cls_mask = (labels == cls) & (umask == 1)
                cls_features = feat[cls_mask.squeeze()] 
                cls_preds = preds[cls_mask.squeeze()] 
                num_samples = cls_features.size(0)
                if num_samples == 0:  
                    center = torch.zeros(feature_dim, device=self.device)
                    cls_centers.append(center)
                    continue
                elif num_samples == 1: 
                    center = cls_features[0] 
                    cls_centers.append(center)
                    continue

                with torch.no_grad():
                    if cls == 1: 
                        sorted_indices = torch.argsort(cls_preds.squeeze(), descending=True)
                    else: 
                        sorted_indices = torch.argsort(cls_preds.squeeze(), descending=False)

                    if sorted_indices.dim() == 0:
                        sorted_indices = sorted_indices.unsqueeze(0) 

                    top_k_num = max(1, int(num_samples * top_k))
                    top_k_indices = sorted_indices[:top_k_num]
                    selected_features = cls_features[top_k_indices]

                    center = selected_features.mean(dim=0)
                    cls_centers.append(center)
            centers[modality] = torch.stack(cls_centers, dim=0)

        return centers
    def compute_modality_wise_centers_iemocap(self, shared_A, shared_T, shared_V, labels, preds, umask, top_k=0.5, num_classes=4):

        centers = {'A': None, 'T': None, 'V': None}
        modalities = {'A': shared_A, 'T': shared_T, 'V': shared_V}

        for modality, feat in modalities.items():
            feature_dim = feat.size(-1)
            cls_centers = []
            for cls in range(num_classes):  
                cls_mask = (labels == cls) & (umask == 1)
                cls_features = feat[cls_mask.squeeze()] 
                cls_preds = preds[cls_mask.squeeze(), cls]  

                num_samples = cls_features.size(0)
                if num_samples == 0: 
                    center = torch.zeros(feature_dim, device=self.device)
                    cls_centers.append(center)
                    continue
                elif num_samples == 1: 
                    center = cls_features[0]
                    cls_centers.append(center)
                    continue

                with torch.no_grad():
                    sorted_indices = torch.argsort(cls_preds, descending=True)

                    if sorted_indices.dim() == 0:
                        sorted_indices = sorted_indices.unsqueeze(0)

                    top_k_num = max(1, int(num_samples * top_k)) 
                    top_k_indices = sorted_indices[:top_k_num]
                    selected_features = cls_features[top_k_indices]

                    center = selected_features.mean(dim=0)
                    cls_centers.append(center)

            centers[modality] = torch.stack(cls_centers, dim=0)

        return centers
    def compute_global_centers(self, batch_centers):
        modality_keys = ['A', 'T', 'V']
        self.batch_count += 1 
        if self.batch_count <= self.warmup_steps:
            momentum = self.start_momentum + (self.ema_momentum - self.start_momentum) * (self.batch_count / self.warmup_steps)
        else:
            momentum = self.ema_momentum

        with torch.no_grad():
            for mod in modality_keys:
                if self.global_centers[mod] is None:
                    self.global_centers[mod] = batch_centers[mod].detach().clone().to(self.device)
                else:
                    self.global_centers[mod] = (
                        momentum * self.global_centers[mod] +
                        (1 - momentum) * batch_centers[mod]
                    ).to(self.device)

        return self.global_centers
    

    def compute_soft_cl_loss(self, features, labels, umask, global_centers, temperature, num_classes=2):
       pass

    def forward(self, shared_A, shared_T, shared_V, temperature, uumask, labels, preds, top_k=0.5):
        """
        shared_A -> [batch, seq_len, dim] or [N, dim]
        shared_T -> [batch, seq_len, dim] or [N, dim]
        shared_V -> [batch, seq_len, dim] or [N, dim]
        preds -> [batch, seq_len] or [N]
        labels -> [batch, seq_len] or [N]
        uumask -> [batch, seq_len] or [N]
        temperature -> scalar, temperature parameter for scaling similarities
        """
        # Reshape inputs
        umask = uumask.reshape(-1, 1) 
        labels = labels.reshape(-1, 1)
        shared_A = shared_A.reshape(-1, shared_A.size(-1))
        shared_T = shared_T.reshape(-1, shared_T.size(-1)) 
        shared_V = shared_V.reshape(-1, shared_V.size(-1)) 
        
        if self.dataset in ['CMUMOSI', 'CMUMOSEI']:
            preds = preds.reshape(-1, 1) # [N, 1]
            labels = labels.long()
            umask[labels == 0] = 0
            labels = torch.where(labels > 0, torch.tensor(1, dtype=torch.long, device=self.device), labels)
            labels = torch.where(labels < 0, torch.tensor(0, dtype=torch.long, device=self.device), labels)
            batch_centers = self.compute_modality_wise_centers(shared_A, shared_T, shared_V, labels, preds, umask, top_k, num_classes=2)
            global_centers = self.compute_global_centers(batch_centers)  # [2, dim]
            features = {'A': shared_A, 'T': shared_T, 'V': shared_V}
            loss = self.compute_soft_cl_loss(features, labels, umask, global_centers, temperature, num_classes=2)
        elif self.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            preds = preds.reshape(-1, preds.size(-1))
            labels = labels.long()
            batch_centers = self.compute_modality_wise_centers_iemocap(shared_A, shared_T, shared_V, labels, preds, umask, top_k, num_classes=4)
            global_centers = self.compute_global_centers(batch_centers)  # [4, dim]
            features = {'A': shared_A, 'T': shared_T, 'V': shared_V}
            loss = self.compute_soft_cl_loss(features, labels, umask, global_centers, temperature, num_classes=4)
        return loss

class MaskedRecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, rec_feats, origi_feats, umask, test_condition=None):
        B, S = umask.shape
        losses = []
        modality_keys = ['a', 't', 'v']
        for i, (rec, orig) in enumerate(zip(rec_feats, origi_feats)):
            if rec.shape[:2] != (B, S) or orig.shape[:2] != (B, S):
                raise ValueError(f"Shape mismatch: rec {rec.shape}, orig {orig.shape}, umask {umask.shape}")
            if test_condition == None:
                rec = F.normalize(rec, p=2, dim=-1)
                orig = F.normalize(orig, p=2, dim=-1) 
            else:
                if modality_keys[i] in test_condition:
                    continue
                else:
                    rec = F.normalize(rec, p=2, dim=-1)
                    orig = F.normalize(orig, p=2, dim=-1)
                    rec = rec.unsqueeze(2)
            n_heads = rec.shape[2]
            head_losses = []
            for h in range(n_heads):
                rec_head = rec[:, :, h, :]
                mse = self.mse_loss(rec_head, orig.detach()) 
                masked_mse = mse * umask.unsqueeze(-1)
                valid_positions = umask.sum()
                if valid_positions > 0:
                    loss = masked_mse.sum() / (valid_positions * rec.shape[-1]) 
                    head_losses.append(loss)
    
            if head_losses:
                modality_loss = sum(head_losses) / len(head_losses)
                losses.append(modality_loss)
        
        return sum(losses)/len(losses) if losses else torch.tensor(0.0, device=umask.device)
