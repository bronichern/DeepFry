import pandas as pd
import os
import os.path
import random
import math
import numpy as np
import torch
import torch.utils.data as data
import soundfile
import glob
from utils import phn_dict, padd_list_tensors
from data.process_data import get_file_labels_allstar, SR


def custom_dataset(window_size, data_path):
    dataset = []
    max_len = 100

    features_files_list = glob.glob(
        os.path.join(data_path, data_path) + "/*.TextGrid")
    for textgrid_file in features_files_list:
        wav_filename = textgrid_file.replace(".TextGrid", ".wav")
        y, _ = soundfile.read(wav_filename)
        
        file_et = len(y)/SR
        labels_count = math.floor(file_et / window_size)
        creaky_labels = np.ones(labels_count)

        for i, idx in enumerate(range(int(np.ceil(len(y)/max_len))-1)):
            start_idx = idx*max_len
            start_y = int(start_idx * (SR * window_size))
            features_idx = start_idx + max_len
            end_y = min(int((features_idx) * (SR * window_size)), len(y))
            creaky_label = creaky_labels[start_idx:features_idx]
            dataset.append([y[start_y: end_y], creaky_label, textgrid_file, start_idx])
    return dataset


def multi_dataset(window_size, data_path):
    dataset = []
    max_len = 100
    features_files_list = glob.glob(
        os.path.join(data_path, data_path) + "/*.TextGrid")
    for textgrid_file in features_files_list:
        wav_filename = textgrid_file.replace(".TextGrid", ".wav")
        creaky_labels_array, voice_labels_array = get_file_labels_allstar(
            textgrid_file, phn_dict, window_size)

        y, _ = soundfile.read(wav_filename)

        labeled_creaky_array_idx = np.array(
            [i for i, f in enumerate(creaky_labels_array)])
        df = pd.DataFrame({'labels': creaky_labels_array,
                          'indexes': labeled_creaky_array_idx})
        df.indexes = df.indexes.astype(int)
        df['groupid'] = (df.labels != df.labels.shift()
                         ).astype(int).fillna(0).cumsum()

        labeled_creaky_array_idx = np.array(
            [i for i, f in enumerate(creaky_labels_array)])
        df = pd.DataFrame({'labels': creaky_labels_array,
                            'indexes': labeled_creaky_array_idx})
        df.indexes = df.indexes.astype(int)
        df = df[df.labels != -1]
        df['groupid'] = df.indexes.diff().fillna(1.0)
        # removed values when filtering -1 - re-index
        df['rindexes'] = list(range(df.shape[0]))
        newgrp = np.zeros(df.shape[0])
        cgrp = 0
        startid = 0
        for (i, r) in df[df.groupid != 1].iterrows():
            newgrp[startid:int(r['rindexes'])] = cgrp
            cgrp += 1
            startid = int(r.rindexes)
        newgrp[startid:] = cgrp
        df['groupid'] = newgrp
        for groupid in df['groupid'].unique():
            curr_group = df[df.groupid == groupid]
            start_idx = curr_group.indexes.iloc[0]
            features_idx = curr_group.indexes.iloc[-1]+1

            if curr_group.shape[0] > max_len:
                slice_size = features_idx - start_idx
                div_val = math.ceil(slice_size/max_len)
                jump_size = slice_size//div_val
                for i in range(1, div_val):
                    half_idx = start_idx + jump_size
                    start_y = int(start_idx * (SR * window_size))
                    end_y = min(
                        int((half_idx) * (SR * window_size)), len(y))

                    creaky_label = creaky_labels_array[start_idx:half_idx]

                    dataset.append(
                        [y[start_y: end_y], creaky_label, textgrid_file, start_idx])
                    assert half_idx > start_idx or start_y < end_y
                    start_idx = half_idx

            start_y = int(start_idx * (SR * window_size))
            end_y = min(int(features_idx * (SR * window_size)), len(y))
            assert features_idx > start_idx or start_y < end_y
            creaky_label = creaky_labels_array[start_idx:features_idx]
            dataset.append([y[start_y: end_y], creaky_label,
                            textgrid_file, start_idx])

    return dataset


class MutiTaskDataset(data.Dataset):
    """
    Dataset for identyifying creak of varying length
    """

    def __init__(self, data_path, seed, window_size, is_custom, normalize=True, augment=False):
        np.random.seed(seed)
        random.seed(seed)
        if is_custom:
            dataset = custom_dataset(window_size, data_path)
        else:
            dataset = multi_dataset(window_size, data_path)
        self.dataset = dataset
        self.normalize = normalize
        self.augment = augment
        print("dataset length:{}".format(len(dataset)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        y, creaky_label, filename, features_idx = self.dataset[index]

        if self.normalize:
            mean_y, std_y = y.mean(), y.std()
            y -= mean_y
            y /= std_y

        y_tensor = torch.FloatTensor(y)
        creaky_labels_tensor = torch.FloatTensor(creaky_label)

        return y_tensor, creaky_labels_tensor, len(creaky_labels_tensor), filename, features_idx

    def __len__(self):
        return len(self.dataset)


class PadCollateRaw:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, predict=False, wav_len=None):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.predict = predict
        self.wav_len = wav_len

    def pad_collate(self, batch):
        """
        args:
            batch - y_tensor, labels_tensor, len(labels_array)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
            non-padded length of x
            name of file example was taken from
            start index of example in file
        """
        # find longest sequence
        wav = [x[0] for x in batch]
        wav_len = [x[0].size(0) for x in batch]
        wav = padd_list_tensors(wav, wav_len, dim=self.dim)

        creaky_target = [x[1] for x in batch]
        target_len = [x[2] for x in batch]
        creaky_target = padd_list_tensors(creaky_target, target_len, self.dim)
        filename = [x[3] for x in batch]
        start = [x[4] for x in batch]
        return wav.unsqueeze(1), creaky_target, target_len, filename, start

    def __call__(self, batch):
        return self.pad_collate(batch)
