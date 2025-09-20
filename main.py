import time
import random
import argparse
import numpy as np
import torch
from Utils.utils import get_loaders, build_model, generate_mask, generate_inputs
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
from Utils.loss import MaskedCELoss, MaskedMSELoss, MaskedRecLoss, MaskedBaseCLLoss, MaskedSoftCLLoss
import sys
sys.path.append('./')
import os
import warnings
warnings.filterwarnings("ignore")
import config
def train_or_eval_model(args, model, reg_loss, cls_loss, softcon_loss, rec_loss, dataloader, optimizer=None, train=False, first_stage=True, mark='train'):
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    losses = []
    batch_s = []
    dataset = args.dataset
    cuda = torch.cuda.is_available() and not args.no_cuda

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(dataloader):
        if train: optimizer.zero_grad()
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        vidname:list:[]
        """
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        vidname = data[-1]
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage) # [seqlen*batch, seqlen*batch, seqlen*batch]
        audio_host_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_host_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_host_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        # [seq, batch, 1] => [batch, seq, 1]
        audio_host_mask = torch.LongTensor(audio_host_mask.transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(text_host_mask.transpose(1, 0, 2)) 
        visual_host_mask = torch.LongTensor(visual_host_mask.transpose(1, 0, 2))
        # guest mask
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage) # [seqlen*batch, view_num]
        audio_guest_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_guest_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_guest_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask.transpose(1, 0, 2))
        text_guest_mask = torch.LongTensor(text_guest_mask.transpose(1, 0, 2))
        visual_guest_mask = torch.LongTensor(visual_guest_mask.transpose(1, 0, 2))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        ## add cuda for tensor
        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.to(device), audio_host_mask.to(device)
            masked_text_host, text_host_mask = masked_text_host.to(device), text_host_mask.to(device)
            masked_visual_host, visual_host_mask = masked_visual_host.to(device), visual_host_mask.to(device)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(device), audio_guest_mask.to(device)
            masked_text_guest, text_guest_mask = masked_text_guest.to(device), text_guest_mask.to(device)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(device), visual_guest_mask.to(device)

            qmask = qmask.to(device)
            umask = umask.to(device)
            label = label.to(device)

        ## generate mask_input_features: ? * [seqlen, batch, dim], input_features_mask: ? * [seq_len, batch, 3]
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                                audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)
        (
            shared_feats,
            res_shas,
            rec_feats,
            origi_feats, 
            original_feats,
            generation_feats,
            original_feats_spe,
            generation_feats_spe
        ) = model(masked_input_features[0], input_features_mask[0], umask, first_stage)
        shared_A, shared_T, shared_V = shared_feats
        res_sha_a, res_sha_t, res_sha_v, res_sha = res_shas       
        # Loss
        all_ = res_sha.view(-1, res_sha.size(2)) # [seq*batch, n_classes]
        all_a, all_t, all_v = res_sha_a.view(-1, res_sha_a.size(2)), res_sha_t.view(-1, res_sha_t.size(2)), res_sha_v.view(-1, res_sha_v.size(2))
        label_ = label.view(-1) # [batch,seq] => [batch * seq]
        if dataset in ['CMUMOSI', 'CMUMOSEI']:
            if first_stage:
                loss_a = reg_loss(all_a, label_, umask)
                loss_t = reg_loss(all_t, label_, umask)
                loss_v = reg_loss(all_v, label_, umask)
                loss_all = reg_loss(all_, label_, umask)
                loss_con =  args.softcon_ratio * softcon_loss(shared_A, shared_T, shared_V, args.temperature, umask, label_, all_, top_k=args.topk)
                loss_rec = args.rec_ratio * rec_loss(generation_feats, original_feats, umask, args.test_condition)
            else:
                loss_a = reg_loss(all_a, label_, umask)
                loss_t = reg_loss(all_t, label_, umask)
                loss_v = reg_loss(all_v, label_, umask)
                loss_all = reg_loss(all_, label_, umask)
                loss_rec = args.rec_ratio * (rec_loss(rec_feats, origi_feats, umask) + rec_loss(generation_feats_spe, original_feats_spe, umask, args.test_condition))
                loss_con = torch.tensor(0.0, device=device)
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            if first_stage:
                loss_a = cls_loss(all_a, label_, umask)
                loss_t = cls_loss(all_t, label_, umask)
                loss_v = cls_loss(all_v, label_, umask)
                loss_all = cls_loss(all_, label_, umask)
                loss_con =  args.softcon_ratio * softcon_loss(shared_A, shared_T, shared_V, args.temperature, umask, label_, all_, top_k=args.topk)
                loss_rec = args.rec_ratio * rec_loss(generation_feats, original_feats, umask, args.test_condition)
            else:
                loss_a = cls_loss(all_a, label_, umask)
                loss_t = cls_loss(all_t, label_, umask)
                loss_v = cls_loss(all_v, label_, umask)
                loss_all = cls_loss(all_, label_, umask)
                loss_rec = args.rec_ratio * (rec_loss(rec_feats, origi_feats, umask) + rec_loss(generation_feats_spe, original_feats_spe, umask, args.test_condition))
                loss_con = torch.tensor(0.0, device=device)


        preds_a.append(all_a.detach())
        preds_t.append(all_t.detach())
        preds_v.append(all_v.detach())
        preds.append(all_.detach())
        labels.append(label_.detach())
        masks.append(umask.view(-1).detach())
        batch_s.append(umask.sum().detach())  
        losses.append(loss_all.detach())
        
        if train and first_stage:
            loss_total = loss_con + loss_a + loss_t + loss_v + loss_all + loss_rec
            loss_total.backward()
            optimizer.step()
        if train and not first_stage:
            loss_total = loss_all + loss_rec + loss_a + loss_t + loss_v
            loss_total.backward()
            optimizer.step()
            
                
    assert preds != [] or print('Error: Preds are Empty')
    preds = torch.cat(preds, dim=0) # [all_batches, n_classes]
    preds_a = torch.cat(preds_a, dim=0)
    preds_t = torch.cat(preds_t, dim=0)
    preds_v = torch.cat(preds_v, dim=0)
    labels = torch.cat(labels, dim=0)
    masks = torch.cat(masks, dim=0)
    
    losses = torch.tensor(losses, device=label_.device)
    batch_s = torch.tensor(batch_s, device=label_.device)
    
    
    #calculate metrics
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        non_zeros = (labels != 0)  # bool tensor on GPU
        mae = torch.mean(torch.abs(labels[non_zeros] - preds[non_zeros].squeeze())).item()
        labels_np = labels[non_zeros].gt(0).cpu().numpy()
        preds_np = preds[non_zeros].squeeze().gt(0).cpu().numpy()
        avg_accuracy = accuracy_score(labels_np, preds_np)
        avg_fscore = f1_score(labels_np, preds_np, average='weighted')
        corr = np.corrcoef(
        labels[non_zeros].squeeze().cpu().numpy(),
        preds[non_zeros].squeeze().cpu().numpy()
        )[0][1]
        avg_acc_a = accuracy_score(
        labels_np, preds_a[non_zeros].squeeze().gt(0).cpu().numpy()
        )
        avg_acc_t = accuracy_score(
        labels_np, preds_t[non_zeros].squeeze().gt(0).cpu().numpy()
        )
        avg_acc_v = accuracy_score(
        labels_np, preds_v[non_zeros].squeeze().gt(0).cpu().numpy()
        )

        avg_loss_rec = round((losses * batch_s).sum().item() / batch_s.sum().item(), 4)
        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], avg_loss_rec
    
    elif dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = preds.argmax(dim=1)      
        preds_a = preds_a.argmax(dim=1)  
        preds_t = preds_t.argmax(dim=1)  
        preds_v = preds_v.argmax(dim=1)  
        preds = preds.detach().cpu().numpy()
        preds_a = preds_a.detach().cpu().numpy()
        preds_t = preds_t.detach().cpu().numpy()
        preds_v = preds_v.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()

        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
        mae = 0 
        ua = recall_score(labels, preds, sample_weight=masks, average='macro')

        avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
        avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
        avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
        avg_loss_rec = round((losses * batch_s).sum().item() / batch_s.sum().item(), 4)
        
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], avg_loss_rec



def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args, cuda, device, save_folder_name, trial=None):
    # load data
    print('=' * 20, 'loading data', '=' * 20)
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature  # feature的

    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    print(audio_root)
    print(text_root)
    print(video_root)

    train_loaders, valid_loaders, test_loaders, adim, tdim, vdim = get_loaders(audio_root=audio_root,
                                                                               text_root=text_root,
                                                                               video_root=video_root,
                                                                               num_folder=args.num_folder,
                                                                               batch_size=args.batch_size,
                                                                               dataset=args.dataset,
                                                                               num_workers=2)
    print(len(train_loaders), len(test_loaders))
    assert len(train_loaders) == args.num_folder, f'Error: folder number'
        
    print('='*20, 'start Training', '='*20)
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_model = []
    folder_acc_as = []
    folder_acc_ts = []
    folder_acc_vs = []
    print(f'{args.num_folder} Cross-validation')
    for ii in range(args.num_folder):
        print(f'>>>>> Cross-validation: training on the {ii + 1} folder >>>>>')
        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print('=' * 80)

        print(f'Step1: build model (each folder has its own model)')
        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss() 
        cls_loss = MaskedCELoss() 
        rec_loss = MaskedRecLoss()
        basecon_loss = MaskedBaseCLLoss(margin=0.1, dataset=args.dataset, device=device)
        softcon_loss = MaskedSoftCLLoss(margin=0.1, dataset=args.dataset, device=device, ema_momentum=0.99, warmup_steps=args.warmup_steps, start_momentum=0.5)
        if cuda:
            model.to(device)
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])
        print('-' * 80)
        print(f'Step2: training (multiple epoches)')
        train_fscores, train_acc_as, train_acc_ts, train_acc_vs, train_accs = [], [], [], [], []
        test_acc_as, test_acc_ts, test_acc_vs = [], [], []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], [] # MAE is mean-absolute-error
        train_loss_recs = []
        models = []
        start_first_stage_time = time.time()
        print("------- Starting the first stage! -------")
        for epoch in range(args.epochs):
            first_stage = True if epoch < args.stage_epoch else False
            
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_rec_loss = train_or_eval_model(args, model, reg_loss, cls_loss, basecon_loss, softcon_loss, rec_loss, train_loader, \
            optimizer=optimizer, train=True, first_stage=first_stage, mark='train')
            
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, _ = train_or_eval_model(args, model, reg_loss, cls_loss, basecon_loss, softcon_loss, rec_loss, test_loader, \
            optimizer=None, train=False, first_stage=first_stage, mark='test')
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)
            test_acc_as.append(test_acc_atv[0])
            test_acc_ts.append(test_acc_atv[1])
            test_acc_vs.append(test_acc_atv[2])
            train_fscores.append(train_fscore)
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])     
            train_loss_recs.append(train_rec_loss)
            models.append(model)
            
            if first_stage:
                print(f'epoch:{epoch}; a_acc_train:{train_acc_atv[0]:.3f}; t_acc_train:{train_acc_atv[1]:.3f}; v_acc_train:{train_acc_atv[2]:.3f}')
                print(f'epoch:{epoch}; a_acc_test:{test_acc_atv[0]:.3f}; t_acc_test:{test_acc_atv[1]:.3f}; v_acc_test:{test_acc_atv[2]:.3f}')
                if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
                    print(f'train_fscore:{train_fscore};test_fscore:{test_fscore}')
                else:
                    print(f'train_acc:{train_acc};test_acc:{test_acc}')
            else:
                if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
                    print(f'epoch:{epoch}; train_fscore_{args.test_condition}:{train_fscore:2.2%}; train_acc_{args.test_condition}:{train_acc:2.2%}')
                    
                    print(f'epoch:{epoch}; test_fscore_{args.test_condition}:{test_fscore:2.2%}; test_acc_{args.test_condition}:{test_acc:2.2%}')
                else:
                    print(f'epoch:{epoch}; train_ua_{args.test_condition}:{train_corr:2.2%}; train_acc_{args.test_condition}:{train_acc:2.2%}')
                    
                    print(f'epoch:{epoch}; test_ua_{args.test_condition}:{test_corr:2.2%}; test_acc_{args.test_condition}:{test_acc:2.2%}')
                
            print('-'*10)
            ## update the parameter for the 2nd stage
            if epoch == args.stage_epoch - 1:
                model = models[-1]

                model_idx_a = int(torch.argmax(torch.Tensor(train_acc_as)))
                print(f'best_epoch_a: {model_idx_a}')
                model_a = models[model_idx_a]
                transformer_a_para_dict = {k: v for k, v in model_a.state_dict().items() if 'Transformer_a' in k}
                model.state_dict().update(transformer_a_para_dict)
                
                model_idx_t = int(torch.argmax(torch.Tensor(train_acc_ts)))
                print(f'best_epoch_t: {model_idx_t}')
                model_t = models[model_idx_t]
                transformer_t_para_dict = {k: v for k, v in model_t.state_dict().items() if 'Transformer_t' in k}
                model.state_dict().update(transformer_t_para_dict)

                model_idx_v = int(torch.argmax(torch.Tensor(train_acc_vs)))
                print(f'best_epoch_v: {model_idx_v}')
                model_v = models[model_idx_v]
                transformer_v_para_dict = {k: v for k, v in model_v.state_dict().items() if 'Transformer_v' in k}
                model.state_dict().update(transformer_v_para_dict)
                
                model_idx_atv = int(torch.argmax(torch.Tensor(train_fscores)))
                print(f'best_epoch_atv: {model_idx_atv}')
                model_atv = models[model_idx_atv]
                transformer_atv_para_dict = {k: v for k, v in model_atv.state_dict().items() if 'cross_attn' in k}
                model.state_dict().update(transformer_atv_para_dict)
                
                model_idx_gen = int(torch.argmin(torch.Tensor(train_loss_recs)))
                print(f'best_epoch_gen: {model_idx_gen}')
                model_gen = models[model_idx_gen]
                # Update to include all specified generation-related modules
                generation_para_dict = {
                    k: v for k, v in model_gen.state_dict().items() 
                    if any(module in k for module in ['a2t_sha', 'a2v_sha', 't2a_sha', 't2v_sha', 'v2a_sha', 'v2t_sha' 'gate_a_sha', 'gate_t_sha', 'gate_v_sha'])
                }
                model.state_dict().update(generation_para_dict)
                
                end_first_stage_time = time.time()
                print("------- Starting the second stage! -------")

        end_second_stage_time = time.time()
        print("-"*80)
        print(f"Time of first stage: {end_first_stage_time - start_first_stage_time}s")
        print(f"Time of second stage: {end_second_stage_time - end_first_stage_time}s")
        print("-"*80)

        print(f'Step3: saving and testing on the {ii+1} folder')
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            best_index_test = np.argmax(np.array(test_fscores[int(args.stage_epoch):])) + int(args.stage_epoch)
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            best_index_test = np.argmax(np.array(test_accs[int(args.stage_epoch):])) + int(args.stage_epoch)
        print(f"Final best is {best_index_test}")
        
        bestmae = test_maes[best_index_test]
        bestcorr = test_corrs[best_index_test]
        bestf1 = test_fscores[best_index_test]
        bestacc = test_accs[best_index_test]
        bestmodel = models[best_index_test]

        folder_mae.append(bestmae)
        folder_corr.append(bestcorr)
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_model.append(bestmodel)
        folder_acc_as.append(max(test_acc_as))
        folder_acc_ts.append(max(test_acc_ts))
        folder_acc_vs.append(max(test_acc_vs))
                
        end_time = time.time()
    print('-'*80)
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        print(f"test_condition ({args.test_condition}) --test_mae {np.mean(folder_mae)} --test_corr {np.mean(folder_corr)} --test_fscores {np.mean(folder_f1)} --test_acc{np.mean(folder_acc)}")
    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        print(f"test_condition ({args.test_condition}) --test_acc {np.mean(folder_acc)} --test_ua {np.mean(folder_corr)}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    ## Params for model
    parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--depth_sha', type=int, default=4, help='depth for shared encoder')
    parser.add_argument('--n_heads', type=int, default=4, help='num_heads for multi-head dropout in specific encoder')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop_rate for all the dropout layers')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='drop_out rate in weighted matrix in Attention')
    parser.add_argument('--mlp_ratio', type=int, default=1, help='the ratio between the MLP hidden size and the multi-head attention hidden size')
    parser.add_argument('--rec_mlp_ratio', type=int, default=1, help='the ratio for rec mlp ratio')
    parser.add_argument('--D_e', type=int, default=128, help='hidden size in attention encoder')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')
    parser.add_argument('--rec_ratio', type=float, default=200, help='ratio for rec_loss')
    parser.add_argument('--rec_loss_ratio', type=float, default=5, help='ratio for loss_rec for calculating drops')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for contrastive loss')
    parser.add_argument('--drop_ratio', type=float, default=1, help='drop ratios for weights to drop')
    parser.add_argument('--warmup_steps', type=int, default=100, help='warmup steps for soft contrastive loss')
    parser.add_argument('--topk', type=float, default=0.5, help='topk')
    parser.add_argument('--basecon_ratio', type=float, default=0.5, help='ratio for base contrastive loss')
    parser.add_argument('--softcon_ratio', type=float, default=0.5, help='ratio for soft contrastive loss')
    
    ## Params for training
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=66, help='make split manner is same with same seed')
    parser.add_argument('--test_condition', type=str, default='atv', choices=['a', 't', 'v', 'at', 'av', 'tv', 'atv'], help='test conditions')
    parser.add_argument('--stage_epoch', type=int, default=150, help='number of epochs of the first stage')
    ## model choose
    parser.add_argument('--model', type=str, default='MMU', help='model to use')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    save_folder_name = f'{args.dataset}'

    ## dataset
    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    
    cuda = torch.cuda.is_available() and not args.no_cuda
    seed_torch(args.seed)
    train(args, cuda, device, save_folder_name)
    
