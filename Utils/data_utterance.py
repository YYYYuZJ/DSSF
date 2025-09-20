import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def read_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        name2feats, feature_dim = pickle.load(f)
    
    print(f'Loaded features from {pkl_path}; dim is {feature_dim}; No. sample is {len(name2feats)}')
    return name2feats, feature_dim

class CMUMOSIDataset(Dataset):
    def __init__(self, label_path, audio_root, text_root, video_root):
        # Read utterance feats
        name2audio, adim = read_data(audio_root)
        name2text, tdim = read_data(text_root)
        name2video, vdim = read_data(video_root)
        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim

        # Initialize data structures
        self.max_len = -1
        self.videoAudioHost = {}
        self.videoTextHost = {}
        self.videoVisualHost = {}
        self.videoAudioGuest = {}
        self.videoTextGuest = {}
        self.videoVisualGuest = {}
        self.videoLabelsNew = {}
        self.videoSpeakersNew = {}
        self.videoIDs, self.videoLabels, self.videoSpeakers, self.videoSentences, self.trainVids, self.valVids, self.testVids = pickle.load(open(label_path, "rb"), encoding='latin1')

        self.vids = sorted(self.trainVids) + sorted(self.valVids) + sorted(self.testVids)

        # Pre-allocate arrays for each video
        for vid in sorted(self.videoIDs):
            uids = self.videoIDs[vid]
            labels = self.videoLabels[vid]
            speakers = self.videoSpeakers[vid]
            seq_len = len(uids)
            self.max_len = max(self.max_len, seq_len)

            speakermap = {'': 0}
            # Pre-allocate numpy arrays
            self.videoAudioHost[vid] = np.zeros((seq_len, self.adim), dtype=np.float32)
            self.videoTextHost[vid] = np.zeros((seq_len, self.tdim), dtype=np.float32)
            self.videoVisualHost[vid] = np.zeros((seq_len, self.vdim), dtype=np.float32)
            self.videoAudioGuest[vid] = np.zeros((seq_len, self.adim), dtype=np.float32)
            self.videoTextGuest[vid] = np.zeros((seq_len, self.tdim), dtype=np.float32)
            self.videoVisualGuest[vid] = np.zeros((seq_len, self.vdim), dtype=np.float32)
            self.videoLabelsNew[vid] = np.array(labels, dtype=np.float32)
            self.videoSpeakersNew[vid] = np.zeros(seq_len, dtype=np.float32)

            # Fill arrays directly
            for ii, uid in enumerate(uids):
                self.videoAudioHost[vid][ii] = name2audio[uid]
                self.videoTextHost[vid][ii] = name2text[uid]
                self.videoVisualHost[vid][ii] = name2video[uid]
                self.videoSpeakersNew[vid][ii] = speakermap[speakers[ii]]

    def __getitem__(self, index):
        vid = self.vids[index]
        return torch.FloatTensor(self.videoAudioHost[vid]),\
               torch.FloatTensor(self.videoTextHost[vid]),\
               torch.FloatTensor(self.videoVisualHost[vid]),\
               torch.FloatTensor(self.videoAudioGuest[vid]),\
               torch.FloatTensor(self.videoTextGuest[vid]),\
               torch.FloatTensor(self.videoVisualGuest[vid]),\
               torch.FloatTensor(self.videoSpeakersNew[vid]),\
               torch.FloatTensor([1]*len(self.videoLabelsNew[vid])),\
               torch.FloatTensor(self.videoLabelsNew[vid]),\
               vid

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print(f'max seqlen: {self.max_len}')
        return self.max_len

    def collate_fn(self, data):
        datnew = []
        dat = pd.DataFrame(data)
        for i in dat:  # column index
            if i <= 5:
                datnew.append(pad_sequence(dat[i]))  # pad
            elif i <= 8:
                datnew.append(pad_sequence(dat[i], True))  # reverse
            else:
                datnew.append(dat[i].tolist())  # origin
        return datnew
    
    
class IEMOCAPDataset(Dataset):
    def __init__(self, label_path, audio_root, text_root, video_root):
        # Read utterance feats
        name2audio, adim = read_data(audio_root)
        name2text, tdim = read_data(text_root)
        name2video, vdim = read_data(video_root)
        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim

        # Initialize data structures
        self.max_len = -1
        self.videoAudioHost = {}
        self.videoTextHost = {}
        self.videoVisualHost = {}
        self.videoAudioGuest = {}
        self.videoTextGuest = {}
        self.videoVisualGuest = {}
        self.videoLabelsNew = {}
        self.videoSpeakersNew = {}
        speakermap = {'F': 0, 'M': 1}
        self.videoIDs, self.videoLabels, self.videoSpeakers, self.videoSentences, self.trainVid, self.testVid = pickle.load(open(label_path, "rb"), encoding='latin1')

        self.vids = sorted(list(self.trainVid | self.testVid))

        # Pre-allocate arrays for each video
        for vid in self.vids:
            uids = self.videoIDs[vid]
            labels = self.videoLabels[vid]
            speakers = self.videoSpeakers[vid]
            seq_len = len(uids)
            self.max_len = max(self.max_len, seq_len)

            # Pre-allocate numpy arrays
            self.videoAudioHost[vid] = np.zeros((seq_len, self.adim), dtype=np.float32)
            self.videoTextHost[vid] = np.zeros((seq_len, self.tdim), dtype=np.float32)
            self.videoVisualHost[vid] = np.zeros((seq_len, self.vdim), dtype=np.float32)
            self.videoAudioGuest[vid] = np.zeros((seq_len, self.adim), dtype=np.float32)
            self.videoTextGuest[vid] = np.zeros((seq_len, self.tdim), dtype=np.float32)
            self.videoVisualGuest[vid] = np.zeros((seq_len, self.vdim), dtype=np.float32)
            self.videoLabelsNew[vid] = np.array(labels, dtype=np.float32)
            self.videoSpeakersNew[vid] = np.zeros(seq_len, dtype=np.float32)

            # Fill arrays directly
            for ii, uid in enumerate(uids):
                self.videoAudioHost[vid][ii] = name2audio[uid]['F']
                self.videoTextHost[vid][ii] = name2text[uid]['F']
                self.videoVisualHost[vid][ii] = name2video[uid]['F']
                self.videoAudioGuest[vid][ii] = name2audio[uid]['M']
                self.videoTextGuest[vid][ii] = name2text[uid]['M']
                self.videoVisualGuest[vid][ii] = name2video[uid]['M']
                self.videoSpeakersNew[vid][ii] = speakermap[speakers[ii]]

    def __getitem__(self, index):
        vid = self.vids[index]
        return torch.FloatTensor(self.videoAudioHost[vid]),\
               torch.FloatTensor(self.videoTextHost[vid]),\
               torch.FloatTensor(self.videoVisualHost[vid]),\
               torch.FloatTensor(self.videoAudioGuest[vid]),\
               torch.FloatTensor(self.videoTextGuest[vid]),\
               torch.FloatTensor(self.videoVisualGuest[vid]),\
               torch.FloatTensor(self.videoSpeakersNew[vid]),\
               torch.FloatTensor([1]*len(self.videoLabelsNew[vid])),\
               torch.LongTensor(self.videoLabelsNew[vid]),\
               vid

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print(f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print(f'max seqlen: {self.max_len}')
        return self.max_len

    def collate_fn(self, data):
        datnew = []
        dat = pd.DataFrame(data)
        for i in dat:  # column index
            if i <= 5:
                datnew.append(pad_sequence(dat[i]))  # pad
            elif i <= 8:
                datnew.append(pad_sequence(dat[i], True))  # reverse
            else:
                datnew.append(dat[i].tolist())  # origin
        return datnew