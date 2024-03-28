import os
from typing import List
import torch
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils import data


class Speech2RoboExpDataset(data.Dataset):
    def __init__(self,
                 seq_data: List[dict],
                 subj_name2id_dict: dict,
                 speech_window_size: int) -> None:
        """
        Dataset for speech-to-robot expression generation.
        Args:
            seq_data: sequence data, list of dictionaries containing the following keys:
                'vert_offset_seq': vertex offset sequence, [seq_len, 3, num_vertex]
                'speech_seq': speech feature sequence, [seq_len, speech_dim], speech_dim = 392
                'name': sequence name, str
            subj_name2id_dict: subject name-to-index dictionary
            speech_window_size: sliding window size for speech features
        """
        self.speech_window_size = speech_window_size
        self.seq_data = seq_data
        self.subj_name2id_dict = subj_name2id_dict
        self.subj_one_hot_labels = np.eye(len(self.subj_name2id_dict.keys()))

    def __getitem__(self,
                    item: int) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, int):
        """
        Args:
            item: index
        Returns:
            speech_seq: speech feature sequence, [seq_len, speech_dim], speech_dim = 392
            vert_offset_seq: vertex offset sequence, [seq_len, 3, num_vertex]
            subj_encoding: subject one-hot encoding, [num_subject]
            speech_window_size: sliding window size for speech features
        """
        name = self.seq_data[item]['name']
        subj, _, _, _ = name.split('+')

        speech_seq = self.seq_data[item]['speech_seq']
        vert_offset_seq = self.seq_data[item]['vert_offset_seq']

        subj_encoding = self.subj_one_hot_labels[:, self.subj_name2id_dict[subj]]

        return (torch.FloatTensor(speech_seq), torch.FloatTensor(vert_offset_seq),
                torch.FloatTensor(subj_encoding), self.speech_window_size)

    def __len__(self):
        return len(self.seq_data)


def load_data(filepaths: List[str]) -> List[dict]:
    """
    Load data from pickle files containing paired speech features and vertex offset sequences.
    Args:
        filepaths: list of pickle filepaths
    Returns:
        seq_data: list of dictionaries containing the following keys:
            'vert_offset_seq': vertex offset sequence, [seq_len, 3, num_vertex]
            'speech_seq': speech feature sequence, [seq_len, speech_dim], speech_dim = 392
            'name': sequence name, str
    """
    print('Loading data...')
    seq_data = {}
    for filepath in tqdm(filepaths):
        subj_name = os.path.split(filepath)[-1].split('.')[0].split('+')[0]

        tmp_data = pickle.load(open(filepath, 'rb'))
        for seq, chunk_data in tmp_data[subj_name]['vert_offset_seq'].items():
            seq_data[subj_name + '+' + seq] = dict()
            seq_data[subj_name + '+' + seq]['vert_offset_seq'] = np.asarray(chunk_data)  # [seq_len, 3, num_vertex]
            seq_data[subj_name + '+' + seq]['name'] = subj_name + '+' + seq

        for seq, chunk_data in tmp_data[subj_name]['speech_seq'].items():
            seq_data[subj_name + '+' + seq]['speech_seq'] = np.asarray(chunk_data)       # [seq_len, speech_dim]

    seq_data = list(seq_data.values())

    return seq_data


def collate_fun(batch: List) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
    """
    Collate function for dataloader.
    Args:
        batch: list of data samples, each sample is a tuple of the following:
            speech_seq: speech feature sequence, [seq_len, speech_dim], speech_dim = 392
            vert_offset_seq: vertex offset sequence, [seq_len, 3, num_vertex]
            subj_encoding: subject one-hot encoding, [num_subject]
            speech_window_size: sliding window size for speech features
    Returns:
        window_speech_seq: windowed speech feature sequence, [seq_len, speech_window_size, speech_dim]
        vert_offset_seq: vertex offset sequence, [seq_len, 3, num_vertex]
        subj_seq: repeated subject one-hot encoding, [seq_len, num_subject]
    """
    batch_size = len(batch)
    if batch_size != 1:                             # batch size is the number of sequences, each with the same subject
        raise NotImplementedError('Batch size greater than 1 is not yet supported.')    # TODO: support batch size > 1

    # get data for the first sample since batch size = 1
    speech_seq = batch[0][0]
    vert_offset_seq = batch[0][1]
    seq_len = vert_offset_seq.shape[0]
    subj_seq = batch[0][2].unsqueeze(dim=0).repeat(seq_len, 1)
    speech_window_size = batch[0][3]

    # add zero-padding at the beginning and the end for speech features
    speech_dim = speech_seq.shape[1]
    speech_seq = torch.cat((torch.zeros((speech_window_size // 2, speech_dim)),
                            speech_seq,
                            torch.zeros((speech_window_size // 2, speech_dim))), dim=0)

    # convert to window speech features
    window_speech_seq = torch.zeros((seq_len, speech_window_size, speech_dim), dtype=torch.float32)
    for j in range(0, seq_len):
        window_speech_seq[j, :, :] = speech_seq[j:j + speech_window_size, :]

    return window_speech_seq, vert_offset_seq, subj_seq


def create_dataloader(filepaths: List[str],
                      subj_name2id_filepath: str,
                      speech_window_size: int,
                      num_workers: int = 0,
                      shuffle: bool = True) -> data.DataLoader:
    """
    Create dataloader.
    Args:
        filepaths: list of pickle filepaths
        subj_name2id_filepath: filepath to subject-to-index mapping
        speech_window_size: sliding window size for speech features
        num_workers: number of workers for dataloader
        shuffle: whether to shuffle data
    Returns:
        seq_dataset: dataloader
    """
    # TODO: support batch size > 1
    BATCH_SIZE = 1  # batch size is the number of sequences, each with the same subject

    # load index mappings
    subj_name2id_dict = pickle.load(open(subj_name2id_filepath, 'rb'))

    # load data
    seq_data = load_data(filepaths)

    # create dataset
    seq_dataset = Speech2RoboExpDataset(seq_data, subj_name2id_dict, speech_window_size)

    # create dataloader
    seq_dataloader = data.DataLoader(dataset=seq_dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                                     num_workers=num_workers, pin_memory=True, collate_fn=collate_fun)

    return seq_dataloader
