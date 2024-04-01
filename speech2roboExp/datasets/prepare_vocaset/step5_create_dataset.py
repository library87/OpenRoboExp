from typing import List
import os
import pickle
import random
import glob
import argparse
import numpy as np
import trimesh
from utils.prepare_dataset_utils import organize_paired_data, split_to_chunks

help_msg = """
Step 5 in prepare_dataset.
This preparation combines synchronized audio-visual data and 
produces pkl files for each subject for training.
The audio-visual data should be organized as follows:
    <reorg_voca<fps>_path>
        ├── video
        │   ├── <subj_1>
        │   │   ├── neutral
        │   │   │   ├── level_1
        │   │   │   │   ├── sentence01.npz, sentence01.mp4
        │   │   │   │   ├── sentence02.npz, sentence02.mp4
        │   ├── <subj_2>
        ├── audio
        │   ├── <subj_1>
        │   ├── <subj_2>
The train data will be saved in the following format: .pkl, each file contains a chunk of sequences.
    <reorg_voca<fps>_path>
        ├── <out_root_path>
        │   ├── train
        │   │   ├── <subj_1>+0.pkl
        │   │   ├── <subj_1>+1.pkl
        │   │   ├── <subj_2>+0.pkl
        │   ├── val
        │   ├── test
For each .pkl file, the data is organized in a dictionary format:
    data[subj]['vert_offset_seq'][seq]: the vertex displacements, [num_frames, 3, num_vertex]
    data[subj]['speech_seq'][seq]: the speech features, [num_frames, feat_dim]
    
Example usage:
python step5_create_dataset.py 
--data_root_path /media/user/WD_BLACK/IROS24/Datasets/reorg/voca_25fps
--subj_mesh_path speech2roboExp/datasets/prepare_vocaset/subj_mesh
--out_root_path /media/user/WD_BLACK/IROS24/Datasets/reorg/voca_25fps/org
--num_val_seqs_per_subj 4
--num_test_seqs_per_subj 4
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, required=True,
                        help='Path to the converted VOCA data (25fps).')
    parser.add_argument('--subj_mesh_path', type=str, required=True,
                        help='Subject-specific face mesh directory.')
    parser.add_argument('--out_root_path', type=str, required=True,
                        help='Output dataset directory.')
    parser.add_argument('--num_val_seqs_per_subj', type=int, default=4,
                        help='Number of sequences per subject reserved for validation.')
    parser.add_argument('--num_test_seqs_per_subj', type=int, default=4,
                        help='Number of sequences per subject reserved for testing.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for random shuffling.')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Number of sequences in a chunk.')
    opt = parser.parse_args()
    print(help_msg)

    # set parameters
    data_root_path = opt.data_root_path
    video_root_path = os.path.join(data_root_path, 'video')
    audio_root_path = os.path.join(data_root_path, 'audio')
    subj_mesh_path = opt.subj_mesh_path
    out_root_path = opt.out_root_path
    num_test_seqs_per_subj = opt.num_test_seqs_per_subj
    num_val_seqs_per_subj = opt.num_val_seqs_per_subj
    seed = opt.seed
    chunk_size = opt.chunk_size

    # initialize
    out_train_path = os.path.join(out_root_path, 'train')
    out_val_path = os.path.join(out_root_path, 'val')
    out_test_path = os.path.join(out_root_path, 'test')
    os.makedirs(out_train_path, exist_ok=True)
    os.makedirs(out_val_path, exist_ok=True)
    os.makedirs(out_test_path, exist_ok=True)
    random.seed(seed)
    subj_names = [name for name in os.listdir(video_root_path) if os.path.isdir(os.path.join(video_root_path, name))]

    num_seqs = []
    for subj_name in subj_names:
        num_seqs.append(len(glob.glob(os.path.join(video_root_path, subj_name) + '/*/*/*.mp4')))

    # create subject-to-index mapping
    subj_name2id_dict = {}
    for i in range(len(subj_names)):
        subj_name2id_dict[subj_names[i]] = i
    pickle.dump(subj_name2id_dict, open(os.path.join(out_root_path, 'subj_name2id.pkl'), 'wb'))

    # loop over the subjects
    for subj_name in subj_names:
        print('Creating paired audio-visual data for subject', subj_name, '...')
        motion_data_filepaths = glob.glob(os.path.join(video_root_path, subj_name) + '/*/*/*.npz')

        # class balance for different subjects
        replicate_times = int(max(num_seqs) / len(motion_data_filepaths)) + 1
        motion_data_filepaths = [motion_data_filepaths] * replicate_times
        motion_data_filepaths = sum(motion_data_filepaths, [])
        motion_data_filepaths = motion_data_filepaths[0:max(num_seqs)]

        # split the dataset
        random.shuffle(motion_data_filepaths)
        train_filepaths = motion_data_filepaths[0:(-num_val_seqs_per_subj - num_test_seqs_per_subj)]
        val_filepaths = motion_data_filepaths[(len(motion_data_filepaths) - num_val_seqs_per_subj -
                                               num_test_seqs_per_subj):(len(motion_data_filepaths) -
                                                                        num_test_seqs_per_subj)]
        test_filepaths = motion_data_filepaths[-num_test_seqs_per_subj:len(motion_data_filepaths)]

        # load subject-specific face mesh
        template_mesh = trimesh.load(os.path.join(subj_mesh_path, subj_name + '.obj'), process=False)
        neutral_vert = template_mesh.vertices.T

        # organize train, val, and test data
        organize_and_save_data(train_filepaths, chunk_size, audio_root_path, subj_name, neutral_vert, out_train_path)
        organize_and_save_data(val_filepaths, chunk_size, audio_root_path, subj_name, neutral_vert, out_val_path)
        organize_and_save_data(test_filepaths, chunk_size, audio_root_path, subj_name, neutral_vert, out_test_path)


def organize_and_save_data(filepaths: List[str],
                           chunk_size: int,
                           audio_root_path: str,
                           subj_name: str,
                           neutral_vert: np.ndarray,
                           out_path: str) -> None:
    """
    Organize and save the paired audiovisual data.
    Args:
        filepaths: list of filepaths.
        chunk_size: number of sequences in a chunk.
        audio_root_path: root path to the audio data.
        subj_name: subject name.
        neutral_vert: neutral face mesh vertices.
        out_path: output path.
    """
    chunks = list(split_to_chunks(filepaths, chunk_size))
    for i, chunk in enumerate(chunks):
        print('Data chunk No.', i, '...')
        data = organize_paired_data(audio_root_path, subj_name, neutral_vert, chunk)
        pickle.dump(data, open(os.path.join(out_path, subj_name + '+' + str(i) + '.pkl'), 'wb'))


if __name__ == "__main__":
    main()
