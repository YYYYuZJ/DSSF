
import sys
import torch
from .data_utterance import CMUMOSIDataset, IEMOCAPDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import config
from module.model import MMU
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('./')

def get_loaders(audio_root, text_root, video_root, dataset, batch_size, num_workers = 16, num_folder = 0):
    """
    :param audio_root: path to audio.npys' folder
    :param text_root: path to text.npys' folder
    :param video_root: path to video.npys' folder'
    :param dataset: dataset's name
    :return:
    """
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']: 
        dataset = IEMOCAPDataset(label_path=config.PATH_TO_LABEL[dataset],
                            audio_root=audio_root,
                            text_root=text_root,
                            video_root=video_root
                           )

        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(num_folder):
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii: train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), 
                                      collate_fn=dataset.collate_fn,
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      prefetch_factor=4)
            test_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=SubsetRandomSampler(test_idxs),
                                     collate_fn=dataset.collate_fn,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     prefetch_factor=4)
            train_loaders.append(train_loader)
            valid_loaders = []
            test_loaders.append(test_loader)

        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, valid_loaders, test_loaders, adim, tdim, vdim
    else:
        dataset = CMUMOSIDataset(label_path=config.PATH_TO_LABEL[dataset],
                                audio_root=audio_root,
                                text_root=text_root,
                                video_root=video_root)
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum + valNum))
        test_idxs = list(range(trainNum + valNum, trainNum + valNum + testNum))

        train_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_idxs),
                                collate_fn=dataset.collate_fn,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=4)
        test_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(test_idxs),
                                collate_fn=dataset.collate_fn,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=False,
                                prefetch_factor=4)
        train_loaders = [train_loader]
        valid_loaders = []
        test_loaders = [test_loader]
        
        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, valid_loaders, test_loaders, adim, tdim, vdim



def build_model(args, adim, tdim, vdim):
    model = MMU(args,
                adim, tdim, vdim, D_e=args.D_e,
                n_classes=args.n_classes, depth_sha=args.depth_sha, num_heads_sha=args.n_heads,
                num_heads_spe=args.n_heads,mlp_ratio=args.mlp_ratio, drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                no_cuda=args.no_cuda)
    print("Model have {} parameters in total".format(sum(x.numel() for x in model.parameters())))
    return model

def generate_mask(seqlen, batch, first_stage):
    audio_mask = np.array([1])
    text_mask = np.array([1])
    visual_mask = np.array([1])
    audio_mask = audio_mask.repeat(seqlen * batch)
    text_mask = text_mask.repeat(seqlen * batch)
    visual_mask = visual_mask.repeat(seqlen * batch)
    matrix = [audio_mask, text_mask, visual_mask]
    return matrix

def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) 
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1)
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim)
    select_feat = torch.where(tmask==0, feat1, feat2) 
    input_features.append(select_feat)
    return input_features
