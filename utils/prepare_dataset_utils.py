from typing import List, Dict
import numpy as np
import os
import copy
from tqdm import tqdm


def interpolate_features(input_feat_seq: np.ndarray, input_fps: int, output_fps: int) -> np.ndarray:
    """
    Interpolate features from input fps to output fps
    Args:
        input_feat_seq: [input_seq_len, feat_dim]
        input_fps: input fps
        output_fps: output fps
    Returns:
        output_feat_seq: [output_seq_len, feat_dim]
    """
    feat_dim = input_feat_seq.shape[1]

    input_seq_len = input_feat_seq.shape[0]
    input_seq_len_in_seconds = input_seq_len / float(input_fps)
    output_seq_len = int(input_seq_len_in_seconds * output_fps)
    input_timestamps = np.arange(input_seq_len) / float(input_fps)
    output_timestamps = np.arange(output_seq_len) / float(output_fps)

    output_feat_seq = np.zeros((output_seq_len, feat_dim))
    for feat in range(feat_dim):
        output_feat_seq[:, feat] = np.interp(output_timestamps,
                                             input_timestamps,
                                             input_feat_seq[:, feat])

    return output_feat_seq


def organize_paired_data(audio_root_path: str,
                         subj: str,
                         neutral_vert: np.ndarray,
                         motion_data_filepaths: List[str]) -> Dict:
    """
    Organize the paired audiovisual data for a subject
    Args:
        audio_root_path: the root path to the speech features that are saved in npy files
        subj: the subject name
        neutral_vert: the neutral face vertices, [3, num_vertex]
        motion_data_filepaths: the list of motion data filepaths that are saved in npz files
    Returns:
        data: the organized data in a dictionary format:
            data[subj]['vert_offset_seq'][seq]: the vertex displacements, [num_frames, 3, num_vertex]
            data[subj]['speech_seq'][seq]: the speech features, [num_frames, feat_dim]
    """
    data = {subj: {}}
    data[subj]['vert_offset_seq'] = {}
    data[subj]['speech_seq'] = {}
    for motion_data_filepath in tqdm(motion_data_filepaths):
        emo_class = os.path.split(motion_data_filepath)[0].split('/')[-2]
        emo_intensity = os.path.split(motion_data_filepath)[0].split('/')[-1]
        seg = os.path.split(motion_data_filepath)[1][0:-4]
        seq = emo_class + '+' + emo_intensity + '+' + seg  # as sequence id

        # check if the corresponding audio file (npy) exists
        audio_filepath = os.path.join(audio_root_path, subj, emo_class, emo_intensity, seg + '.npy')
        if not os.path.isfile(audio_filepath):
            print('Corresponding audio npy file does not exist! Skip!')
            continue
        speech_seq = np.load(audio_filepath)     # 25 fps during speech feature extraction

        if not os.path.isfile(motion_data_filepath):
            print('Corresponding motion npz file does not exist! Skip!')
            continue
        f = np.load(motion_data_filepath)
        vert_seq = copy.deepcopy(f['vert_seq'])  # [num_frames, 3, num_vertex]

        # force alignment: simply cut the longer one
        if speech_seq.shape[0] > vert_seq.shape[0]:
            speech_seq = speech_seq[0:vert_seq.shape[0], :]
        else:
            vert_seq = vert_seq[0:speech_seq.shape[0], :, :]

        # append face vertex displacements
        try:
            data[subj]['vert_offset_seq'][seq] = [vert - neutral_vert for vert in vert_seq if vert is not None]
        except TypeError:
            print('Missing frame found! Skip!')
            del data[subj]['vert_offset_seq'][seq]
            continue

        # append audio features
        data[subj]['speech_seq'][seq] = speech_seq

    return copy.deepcopy(data)


def split_to_chunks(input_lst: List,
                    num_chunks: int) -> List:
    """
    Yield successive n-sized chunks from list.
    Args:
        input_lst: input list
        num_chunks: chunk size
    Returns:
        chunks: list of chunks
    """
    for i in range(0, len(input_lst), num_chunks):
        yield input_lst[i:i + num_chunks]
