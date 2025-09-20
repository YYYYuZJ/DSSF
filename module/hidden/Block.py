import torch
from torch import nn
import torch.nn.functional as F

class CosSimLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rec_feat, origi_feat, ratio=5):
        _, B, S, D_e = rec_feat.shape
        rec = F.normalize(rec_feat, p=2, dim=-1)
        orig = F.normalize(origi_feat, p=2, dim=-1)     
        loss = torch.sum(rec*orig, dim=-1) * ratio
        return loss

class RecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, rec_feat, origi_feat, ratio=5):
        n_heads, B, S, D_e = rec_feat.shape
        rec = F.normalize(rec_feat, p=2, dim=-1)
        orig = F.normalize(origi_feat, p=2, dim=-1)     
        orig = orig.unsqueeze(0).repeat(n_heads, 1, 1, 1)
        mse = self.mse_loss(rec, orig)
        loss = mse.sum(dim=-1) * ratio
        return loss

class MLPLayer(nn.Module):
    def __init__(self, 
                in_features, 
                out_features): 
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))
    
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block_softmoe(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0,
            proj_drop=0,
            mlp_ratio=1,
    ):
        super().__init__()
        self.Transformer_a = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_t = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )
        self.Transformer_v = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x, cross_modality='atv', mask_modality=None, mask=None):
        if cross_modality == 'a':
            x_a_mlp = self.Transformer_a(x, mask_modality, mask)
            return x_a_mlp
        if cross_modality == 't':
            x_t_mlp = self.Transformer_t(x, mask_modality, mask)
            return x_t_mlp
        if cross_modality == 'v':
            x_v_mlp = self.Transformer_v(x, mask_modality, mask)
            return x_v_mlp



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )

    def forward(self, x, mask_modality, mask=None):
        B, seq_len, C = x.shape
        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        if mask is not None:
            mask = mask.bool()
            if not (mask_modality == 'atv'):
                mask = {'a':mask[:, :seq_len], 't':mask[:, seq_len:2*seq_len], 'v':mask[:, 2*seq_len:3*seq_len]}
                mask = mask[mask_modality]
            attn = self.attn_drop(attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1).type_as(x))
            attn = torch.where(torch.isnan(attn), torch.full_like(attn, 0), attn)
        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)
        return x_out


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Attention(
                    dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
    def forward(self, x, mask=None, modality=None):
        for layer_idx, block in enumerate(self.blocks):
            x = x + block(x, modality, mask=mask)
        return x


class Block_spe(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            depth=4,
            n_heads=4,
            rec_mlp_ratio=1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.depth = depth
        n_heads = self.n_heads
        self.drop_sha = nn.Dropout(drop)
        self.proj_in = nn.ModuleDict({
            'a': nn.Linear(dim, dim),
            't': nn.Linear(dim, dim),
            'v': nn.Linear(dim, dim)
        })
        self.rec_loss = CosSimLoss()
        self.recs = nn.ModuleList([Mlp(in_features=int(dim + dim//n_heads), hidden_features=int(dim + dim//n_heads) * rec_mlp_ratio, out_features=dim, drop=0) for _ in range(n_heads)]) 
        
        self.blocks = nn.ModuleList(
            [
                Block_softmoe(dim,
                              num_heads=num_heads,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              mlp_ratio=mlp_ratio,)
                for i in range(depth)
            ]
        )
        
    def dynamic_weighting(self, x_in, shared_feature, modality, orig_feat):
        pass
    
    
    def weights(self, x, weights):
        pass
        
    
    def forward(self, x, shared_feature, mask=None, modality=None, first_stage=True, rec_loss_ratio=5, drop_ratio=1):
        if first_stage:
            for layer_idx, block in enumerate(self.blocks):
                x = x + block(x, cross_modality=modality, mask_modality=modality, mask=mask)
            return x
        else: 
            drops,rec_feats = [], []
            B, S, D_e = x.shape
            original_feat = x.clone().detach()
            for layer_idx, block in enumerate(self.blocks):
                rec_feat = torch.zeros([B,S,self.n_heads,D_e], device=x.device)
                drop = torch.zeros([B, S, self.n_heads], device=x.device)
                
                x_out, rec_feat, drop = self.dynamic_weighting(x, shared_feature, modality, orig_feat=original_feat, n_heads=self.n_heads, rec_loss_ratio=rec_loss_ratio, drop_ratio=drop_ratio)    
                
                x = x + x_out
                x = x + block(x, cross_modality=modality, mask_modality=modality, mask=mask)
                rec_feats.append(rec_feat)
                drops.append(drop.detach())
                
            rec = torch.cat(rec_feats, dim=-1).reshape(B, S, self.n_heads, self.depth, D_e).mean(dim=3)
            drops = torch.cat(drops, dim=-1).reshape(B, S, self.depth, self.n_heads).mean(dim=2)
            return x, rec, drops