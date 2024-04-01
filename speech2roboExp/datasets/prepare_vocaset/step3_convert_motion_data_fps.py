import os
from glob import glob
import numpy as np
import trimesh
from tqdm import tqdm
import shutil
import argparse
from utils.prepare_dataset_utils import interpolate_features
from render_utils.render import render_video


help_msg = """
Step 3 in prepare_dataset. Convert the fps of the motion data from 60 to 25.
Preserve the original 60fps voca dataset and create a new dataset with 25fps.
The motion data should be organized as follows:
    <reorg_voca_path>
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
    
The converted motion data will be saved in the following format: (maintain the same structure)
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

Example usage:
python step3_convert_motion_data_fps.py
--src_root_path /media/user/WD_BLACK/IROS24/Datasets/reorg/voca
--dst_root_path /media/user/WD_BLACK/IROS24/Datasets/reorg/voca_25fps
--template_mesh_filepath speech2roboExp/assets/human_voca/ABase.obj
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root_path', type=str, required=True,
                        help='The path to the source voca data. 60fps')
    parser.add_argument('--dst_root_path', type=str, required=True,
                        help='The path to save the converted voca data. 25fps')
    parser.add_argument('--template_mesh_filepath', type=str, required=True,
                        help='The path to the template mesh (zero-shape zero-pose face mesh).')
    opt = parser.parse_args()

    print(help_msg)

    # set parameters
    src_root_path = opt.src_root_path
    dst_root_path = opt.dst_root_path
    template_mesh_filepath = opt.template_mesh_filepath

    input_fps = 60      # the original fps of the 3D motion data
    output_fps = 25     # the target fps of the 3D motion data

    # initialize
    template_mesh = trimesh.load(template_mesh_filepath, process=False)
    faces = template_mesh.faces.astype(np.int32)    # just for rendering

    # loop over the sequences
    src_motion_root_path = os.path.join(src_root_path, 'video')
    subj_names = [x for x in os.listdir(src_motion_root_path) if os.path.isdir(os.path.join(src_motion_root_path, x))]
    for subj_name in tqdm(subj_names):
        src_motion_path = os.path.join(src_motion_root_path, subj_name, 'neutral', 'level_1')
        src_motion_data_filepaths = glob(src_motion_path + '/*.npz')

        for src_motion_data_filepath in tqdm(src_motion_data_filepaths):
            src_audio_filepath = src_motion_data_filepath.replace('video', 'audio').replace('npz', 'wav')
            src_speech_feat_filepath = src_motion_data_filepath.replace('video', 'audio').replace('npz', 'npy')

            # copy audio file and speech feature file
            dst_audio_filepath = src_audio_filepath.replace(src_root_path, dst_root_path)
            os.makedirs(os.path.dirname(dst_audio_filepath), exist_ok=True)
            shutil.copy(src_audio_filepath, dst_audio_filepath)
            dst_speech_feat_filepath = src_speech_feat_filepath.replace(src_root_path, dst_root_path)
            shutil.copy(src_speech_feat_filepath, dst_speech_feat_filepath)

            # interpolate vertex sequence
            f = np.load(src_motion_data_filepath)
            vert_seq = f['vert_seq']    # [seq_len, 3, num_vertex]
            vert_seq = np.reshape(vert_seq, (vert_seq.shape[0], -1), order='F')     # [seq_len, 3 * num_vertex]
            vert_seq = interpolate_features(vert_seq, input_fps=input_fps, output_fps=output_fps)
            vert_seq = np.reshape(vert_seq, (vert_seq.shape[0], 3, -1), order='F')  # [seq_len, 3, num_vertex]

            # save interpolated vertex sequence
            dst_motion_data_filepath = src_motion_data_filepath.replace(src_root_path, dst_root_path)
            os.makedirs(os.path.dirname(dst_motion_data_filepath), exist_ok=True)
            np.savez_compressed(dst_motion_data_filepath, vert_seq=vert_seq)

            # render video
            dst_video_filepath = dst_motion_data_filepath.replace('npz', 'mp4')
            render_video(dst_video_filepath, dst_audio_filepath, vert_seq, faces, fps=output_fps, width=512)


if __name__ == "__main__":
    main()
