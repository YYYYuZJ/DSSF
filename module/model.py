import torch
import torch.nn as nn
import torch.nn.functional as F
from module.hidden.Block import *
class MMU(nn.Module):
    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, depth_sha=4, depth_spe=4, num_heads_sha=2, num_heads_spe=2,
                 mlp_ratio=1, drop_rate=0, attn_drop_rate=0,no_cuda=False):
        print(f"D_e is:{D_e}, num_heads is:{num_heads_sha}, drop_rate is:{drop_rate}, attn_drop is:{attn_drop_rate}")
        super(MMU, self).__init__()
        self.args = args
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads_sha
        drop_rate = args.drop_rate
        n_heads = args.n_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate
        attn_drop_rate = args.attn_drop_rate
        
        self.a2a_sha = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t2t_sha = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v2v_sha = nn.Sequential(nn.Linear(self.vdim, D_e))
        
        self.a2a_spe = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t2t_spe = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v2v_spe = nn.Sequential(nn.Linear(self.vdim, D_e))
        
        self.dropout_a = nn.Dropout(drop_rate)
        self.dropout_t = nn.Dropout(drop_rate)
        self.dropout_v = nn.Dropout(drop_rate)
        
        self.dropout_a_spe = nn.Dropout(drop_rate)
        self.dropout_t_spe = nn.Dropout(drop_rate)
        self.dropout_v_spe = nn.Dropout(drop_rate)
        
        self.enc_sha = Block(
            dim=D_e,
            num_heads=num_heads_sha,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth_sha
        )
        
        self.enc_spe = Block_spe(
            dim=D_e,
            num_heads=num_heads_spe,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth_spe,
            n_heads = n_heads,
            rec_mlp_ratio = args.rec_mlp_ratio,
        )
        
        self.head_1_a = nn.Linear(D_e, n_classes)
        self.head_1_t = nn.Linear(D_e, n_classes)
        self.head_1_v = nn.Linear(D_e, n_classes)
        
        self.head_sha = nn.Linear(D_e, n_classes)
        
        self.head = nn.Linear(4 * D_e, n_classes)
        self.proj = nn.Linear(4 * D_e, 4 * D_e)
        
        self.gate_a_sha = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=1,
            drop=drop_rate,
        )
        self.gate_t_sha = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=1,
            drop=drop_rate,
        )
        self.gate_v_sha = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=1,
            drop=drop_rate,
        )
        
        self.router_spe = Mlp(
            in_features=D,
            hidden_features=int(D * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        
        self.cross_attn = nn.ModuleList(
            [
                Attention(
                    dim=D_e,
                    num_heads=num_heads_sha,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    mlp_ratio=mlp_ratio
                )
                for i in range(depth_sha)
            ]
        )
        
        self.gate_a_spe = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=1,
            drop=drop_rate,
        )
        self.gate_t_spe = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=1,
            drop=drop_rate,
        )
        self.gate_v_spe = Mlp(
            in_features=D_e,
            hidden_features=int(D_e * mlp_ratio),
            out_features=1,
            drop=drop_rate,
        )
        self.a2t_sha = MLPLayer(adim, D_e)
        self.a2v_sha = MLPLayer(adim, D_e)
        self.t2a_sha = MLPLayer(tdim, D_e)
        self.t2v_sha = MLPLayer(tdim, D_e)
        self.v2t_sha = MLPLayer(vdim, D_e)
        self.v2a_sha = MLPLayer(vdim, D_e)
        self.a2t_spe = MLPLayer(adim, D_e)
        self.a2v_spe = MLPLayer(adim, D_e)
        self.t2v_spe = MLPLayer(tdim, D_e)
        self.t2a_spe = MLPLayer(tdim, D_e)
        self.v2a_spe = MLPLayer(vdim, D_e)
        self.v2t_spe = MLPLayer(vdim, D_e)
        self.dropout_rec_a = nn.Dropout(drop_rate)
        self.dropout_rec_t = nn.Dropout(drop_rate)
        self.dropout_rec_v = nn.Dropout(drop_rate)


    def get_feats_sha(self, x_a, x_t, x_v, attn_mask):
        B, S, D_e = x_a.shape
        x = torch.cat([x_a, x_t, x_v], dim=1)
        for layer_idx, cross in enumerate(self.cross_attn):
            x = x + cross(x, mask_modality='atv', mask=attn_mask)
        x[attn_mask==0] = 0
        x_a, x_t, x_v = x[:, :S, :], x[:, S:2*S, :], x[:, 2*S:, :]
        x = torch.cat([x_a, x_t, x_v], dim=-1)
        x = x.reshape(B, S, 3, D_e)
        x = x.mean(dim=2)
        return x, x_a, x_t, x_v
    
    def generate(self, test_condition, x_a, x_t, x_v, spe=False):
        if not spe:
            a2t = self.a2t_sha
            a2v = self.a2v_sha
            t2a = self.t2a_sha
            t2v = self.t2v_sha
            v2a = self.v2a_sha
            v2t = self.v2t_sha
            a2a = self.a2a_sha
            t2t = self.t2t_sha
            v2v = self.v2v_sha
            gate_a = self.gate_a_sha
            gate_t = self.gate_t_sha
            gate_v = self.gate_v_sha
        else:
            a2t = self.a2t_spe
            a2v = self.a2v_spe
            t2a = self.t2a_spe
            t2v = self.t2v_spe
            v2a = self.v2a_spe
            v2t = self.v2t_spe
            a2a = self.a2a_spe
            t2t = self.t2t_spe
            v2v = self.v2v_spe
            gate_a = self.gate_a_spe
            gate_t = self.gate_t_spe
            gate_v = self.gate_v_spe
        
        if test_condition == 'atv':
            x_a_rec, x_t_rec, x_v_rec = a2a(x_a), t2t(x_t), v2v(x_v)
        
        elif test_condition == 'a':
            x_a_rec = a2a(x_a)
            x_v_rec = a2v(x_a.transpose(1,2)).transpose(1,2)
            x_t_rec = a2t(x_a.transpose(1,2)).transpose(1,2)
            x_v_rec = x_a_rec
            x_t_rec = x_a_rec
        elif test_condition == 't':
            x_t_rec = t2t(x_t)
            x_v_rec = t2v(x_t.transpose(1,2)).transpose(1,2)
            x_a_rec = t2a(x_t.transpose(1,2)).transpose(1,2)
            x_v_rec = x_t_rec
            x_a_rec = x_t_rec
        elif test_condition == 'v':
            x_v_rec = v2v(x_v)
            x_t_rec = v2t(x_v.transpose(1,2)).transpose(1,2)
            x_a_rec = v2a(x_v.transpose(1,2)).transpose(1,2)
            x_t_rec = x_v_rec
            x_a_rec = x_v_rec
        
        elif test_condition == 'at':
            x_a_rec = a2a(x_a)
            x_t_rec = t2t(x_t)
            v_from_a = a2v(x_a.transpose(1,2)).transpose(1,2)
            v_from_t = t2v(x_t.transpose(1,2)).transpose(1,2)
            score_a = gate_v(v_from_a)
            score_t = gate_v(v_from_t)
            scores = torch.cat([score_a, score_t], dim=-1)
            weights = F.softmax(scores, dim=-1)
            weight_a, weight_t = weights[:, :, 0:1], weights[:, :, 1:2]
            x_v_rec = weight_a * v_from_a + weight_t * v_from_t
            x_v_rec = (x_a_rec + x_t_rec) / 2
        
        elif test_condition == 'av':
            x_a_rec = a2a(x_a)
            x_v_rec = v2v(x_v)
            t_from_a = a2t(x_a.transpose(1,2)).transpose(1,2) 
            t_from_v = v2t(x_v.transpose(1,2)).transpose(1,2)
            score_a = gate_t(t_from_a)
            score_v = gate_t(t_from_v)
            scores = torch.cat([score_a, score_v], dim=-1)
            weights = F.softmax(scores, dim=-1)
            weight_a, weight_v = weights[:, :, 0:1], weights[:, :, 1:2]
            x_t_rec = weight_a * t_from_a + weight_v * t_from_v
            x_t_rec = (x_a_rec + x_v_rec) / 2
        
        elif test_condition == 'tv':
            x_t_rec = t2t(x_t)
            x_v_rec = v2v(x_v)
            a_from_t = t2a(x_t.transpose(1,2)).transpose(1,2)
            a_from_v = v2a(x_v.transpose(1,2)).transpose(1,2)
            score_t = gate_a(a_from_t)
            score_v = gate_a(a_from_v)
            scores = torch.cat([score_t, score_v], dim=-1)
            weights = F.softmax(scores, dim=-1)
            weight_t, weight_v = weights[:, :, 0:1], weights[:, :, 1:2]
            x_a_rec = weight_t * a_from_t + weight_v * a_from_v
            x_a_rec = (x_t_rec + x_v_rec) / 2
        return x_a_rec, x_t_rec, x_v_rec
       
            
    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False):
        weight_save = []
        
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], inputfeats[:, :,self.adim + self.tdim:]
        seq_len, batch_size, original_dim_a = audio.shape

        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        attn_mask = input_mask.transpose(1, 2).reshape(batch_size, -1)
        
        if first_stage:
            drop_a = drop_t = drop_v = torch.zeros(batch_size, seq_len, self.args.n_heads, device=self.device, requires_grad=False)
            rec_a = rec_t = rec_v = torch.zeros(batch_size, seq_len, self.args.n_heads, self.D_e, device=self.device, requires_grad=False)
            weight_save.append(torch.zeros(batch_size, seq_len, 3))
            proj_a_spe = proj_t_spe = proj_v_spe = torch.zeros(batch_size, seq_len, self.D_e, device=self.device, requires_grad=False)
            
            proj_a_sha = self.a2a_sha(audio)
            proj_t_sha = self.t2t_sha(text)
            proj_v_sha = self.v2v_sha(video)
            original_feats = [self.dropout_rec_a(proj_a_sha.detach().clone()), self.dropout_rec_t(proj_t_sha.detach().clone()), self.dropout_rec_v(proj_v_sha.detach().clone())]

            proj_a_sha = self.dropout_a(proj_a_sha)
            proj_t_sha = self.dropout_t(proj_t_sha)
            proj_v_sha = self.dropout_v(proj_v_sha)
            
            x_a_shared = self.enc_sha(proj_a_sha, attn_mask, 'a')
            x_t_shared = self.enc_sha(proj_t_sha, attn_mask, 't')
            x_v_shared = self.enc_sha(proj_v_sha, attn_mask, 'v')
            
            x_shared = torch.cat([x_a_shared, x_t_shared, x_v_shared], dim=1)
            x_shared[attn_mask == 0] = 0
            x_a_shared, x_t_shared, x_v_shared = x_shared[:, :seq_len, :], x_shared[:, seq_len:2*seq_len, :], x_shared[:, 2*seq_len:, :]
            
            shared_feat, _ , _ , _ =self.get_feats_sha(x_a_shared, x_t_shared, x_v_shared, attn_mask)
            
            out = self.head_sha(shared_feat)
            out_a = self.head_1_a(x_a_shared)
            out_t = self.head_1_t(x_t_shared)
            out_v = self.head_1_v(x_v_shared)
            
            proj_a_sha, proj_t_sha, proj_v_sha = self.generate(self.args.test_condition, audio, text, video, False)
            generation_feats = [self.dropout_rec_a(proj_a_sha.detach().clone()), self.dropout_rec_t(proj_t_sha.detach().clone()), self.dropout_rec_v(proj_v_sha.detach().clone())]
            generation_feats_spe = original_feats_spe = [
                                                         torch.zeros((batch_size, seq_len, self.D_e), device=self.device,requires_grad=False),
                                                         torch.zeros((batch_size, seq_len, self.D_e), device=self.device, requires_grad=False),
                                                         torch.zeros((batch_size, seq_len, self.D_e), device=self.device, requires_grad=False)
                                                        ]            
            
        else:
            generation_feats = original_feats = [
                                                    torch.zeros((batch_size, seq_len, self.D_e), device=self.device,requires_grad=False),
                                                    torch.zeros((batch_size, seq_len, self.D_e), device=self.device, requires_grad=False),
                                                    torch.zeros((batch_size, seq_len, self.D_e), device=self.device, requires_grad=False)
                                                ]          
            with torch.no_grad():
                proj_a_sha, proj_t_sha, proj_v_sha = self.generate(self.args.test_condition, audio, text, video, False)
                proj_a_sha = self.dropout_a(proj_a_sha)
                proj_t_sha = self.dropout_t(proj_t_sha)
                proj_v_sha = self.dropout_v(proj_v_sha)
                x_a_shared = self.enc_sha(proj_a_sha, attn_mask, 'a')
                x_t_shared = self.enc_sha(proj_t_sha, attn_mask, 't')
                x_v_shared = self.enc_sha(proj_v_sha, attn_mask, 'v')
                x_shared = torch.cat([x_a_shared, x_t_shared, x_v_shared], dim=1)
                x_shared[attn_mask == 0] = 0
                x_a_shared, x_t_shared, x_v_shared = x_shared[:, :seq_len, :], x_shared[:, seq_len:2*seq_len, :], x_shared[:, 2*seq_len:, :]
                shared_feat, x_a_shared, x_t_shared, x_v_shared =self.get_feats_sha(x_a_shared, x_t_shared, x_v_shared, attn_mask)
            
            proj_a_spe = self.a2a_spe(audio)
            proj_t_spe = self.t2t_spe(text)
            proj_v_spe = self.v2v_spe(video)
            original_feats_spe = [self.dropout_rec_a(proj_a_spe.detach().clone()), self.dropout_rec_t(proj_t_spe.detach().clone()), self.dropout_rec_v(proj_v_spe.detach().clone())]
            proj_a_spe, proj_t_spe, proj_v_spe = self.generate(self.args.test_condition, audio, text, video, True)
            generation_feats_spe = [self.dropout_rec_a(proj_a_spe.detach().clone()), self.dropout_rec_t(proj_t_spe.detach().clone()), self.dropout_rec_v(proj_v_spe.detach().clone())]
            proj_a_spe = self.dropout_a_spe(proj_a_spe)
            proj_t_spe = self.dropout_t_spe(proj_t_spe)
            proj_v_spe = self.dropout_v_spe(proj_v_spe)
            
            x_a_spe , rec_a, drop_a = self.enc_spe(proj_a_spe, x_a_shared, attn_mask, 'a', first_stage, self.args.rec_loss_ratio, self.args.drop_ratio)
            x_t_spe , rec_t, drop_t = self.enc_spe(proj_t_spe, x_t_shared, attn_mask, 't', first_stage, self.args.rec_loss_ratio, self.args.drop_ratio)
            x_v_spe , rec_v, drop_v = self.enc_spe(proj_v_spe, x_v_shared, attn_mask, 'v', first_stage, self.args.rec_loss_ratio, self.args.drop_ratio)
            
            x = torch.cat([x_a_spe, x_t_spe, x_v_spe], dim=1)
            x[attn_mask == 0] = 0
            x_a_spe, x_t_spe, x_v_spe = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
            
            x = torch.cat([x_a_spe, x_t_spe, x_v_spe], dim=-1)
            weights = self.router_spe(x)
            weights = torch.softmax(weights, dim=-1)
            weight_save.append(weights.clone())

            weights = weights.unsqueeze(-1)
            x_unweighted_spe = x.view(batch_size, seq_len, 3, self.D_e)
            x_spe_out = weights * x_unweighted_spe
            x_spe_out = x_spe_out.view(batch_size, seq_len, 3 * self.D_e)
                                    
            rec_spe = torch.cat([rec_a, rec_t, rec_v], dim=1)
            drop_spe = torch.cat([drop_a, drop_t, drop_v], dim=1)
            rec_spe[attn_mask == 0] = 0
            drop_spe[attn_mask == 0] = 0
            rec_a, rec_t, rec_v = rec_spe[:, :seq_len, :], rec_spe[:, seq_len:2*seq_len, :], rec_spe[:, 2*seq_len:, :]
            drop_a, drop_t, drop_v = drop_spe[:, :seq_len, :], drop_spe[:, seq_len:2*seq_len, :], drop_spe[:, 2*seq_len:, :]
            
            out_a = self.head_1_a(x_a_spe)
            out_t = self.head_1_t(x_t_spe)
            out_v = self.head_1_v(x_v_spe)
            
            res = torch.cat([shared_feat, x_spe_out], dim=-1)
            u = F.relu(self.proj(res))
            u = F.dropout(u, p=self.out_dropout, training=self.training)
            hidden = u + res
            out = self.head(hidden)

        return (
            [x_a_shared, x_t_shared, x_v_shared],
            [out_a, out_t, out_v, out],
            [rec_a, rec_t, rec_v],
            [proj_a_spe, proj_t_spe, proj_v_spe],
            [drop_a, drop_t, drop_v],
            original_feats,
            generation_feats,
            original_feats_spe,
            generation_feats_spe
        )