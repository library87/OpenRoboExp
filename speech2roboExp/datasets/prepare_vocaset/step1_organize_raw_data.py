import os
import trimesh
import numpy as np
from tqdm import tqdm
import shutil
import argparse
from render_utils.render import render_video


help_msg = """
Step 1 in prepare_dataset. Organize voca raw data into standardized format.
The voca raw data should be organized as follows:
    <raw_voca_path>
    ├── unposedcleaneddata
    │   ├── <subj_1>
    │   │   ├── sentence01
    │   │   │   ├── sentence01.000001.ply
    │   │   │   ├── sentence01.000002.ply
    │   │   ├── sentence02
    │   │   │   ├── sentence02.000001.ply
    │   │   │   ├── sentence02.000002.ply
    │   ├── <subj_2>
    ├── audio
    │   ├── <subj_1>
    │   │   ├── sentence01.wav
    │   │   ├── sentence02.wav
    │   ├── <subj_2>
The organized data will be saved in the following format: .npz for 3D motion data and .wav for audio data.
.npz: [vert_seq: np.ndarray] with shape [num_frames, 3, num_vertex]
    <reorg_voca_path>
    ├── video
    │   ├── <subj_1>
    │   │   ├── neutral                 # the emotion label (for voca, only neutral is available)
    │   │   │   ├── level_1             # the level of the emotion
    │   │   │   │   ├── sentence01.npz, sentence01.mp4
    │   │   │   │   ├── sentence02.npz, sentence02.mp4
    │   ├── <subj_2>
    ├── audio
    │   ├── <subj_1>
    │   │   ├── neutral
    │   │   │   ├── level_1
    │   │   │   │   ├── sentence01.wav
    │   │   │   │   ├── sentence02.wav
    │   ├── <subj_2>

Example usage:
python step1_organize_raw_data.py 
--raw_motion_path /media/user/WD_BLACK/IROS24/Datasets/raw/voca/unposedcleaneddata/ 
--raw_audio_path /media/user/WD_BLACK/IROS24/Datasets/raw/voca/audio/
--dst_root_path /media/user/WD_BLACK/IROS24/Datasets/reorg/voca/
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_motion_path', type=str, required=True,
                        help='The path to the raw 3D motion data.')
    parser.add_argument('--raw_audio_path', type=str, required=True,
                        help='The path to the raw audio data.')
    parser.add_argument('--dst_root_path', type=str, required=True,
                        help='The path to the organized VOCA folder (60fps).')
    opt = parser.parse_args()

    print(help_msg)

    # set parameters
    raw_motion_path = opt.raw_motion_path
    raw_audio_path = opt.raw_audio_path
    dst_root_path = opt.dst_root_path

    fps = 60            # the original fps of the 3D motion data
    num_vertex = 5023   # number of vertices in the 3D motion data

    # initialize
    video_root_path = os.path.join(dst_root_path, 'video')
    audio_root_path = os.path.join(dst_root_path, 'audio')

    # loop over the sequences
    subj_names = [x for x in os.listdir(raw_motion_path) if os.path.isdir(os.path.join(raw_motion_path, x))]
    for subj_name in tqdm(subj_names):
        dst_video_path = os.path.join(video_root_path, subj_name, 'neutral', 'level_1')
        os.makedirs(dst_video_path, exist_ok=True)

        dst_audio_path = os.path.join(audio_root_path, subj_name, 'neutral', 'level_1')
        os.makedirs(dst_audio_path, exist_ok=True)

        raw_seq_root_path = os.path.join(raw_motion_path, subj_name)
        seq_names = [x for x in os.listdir(raw_seq_root_path) if
                     os.path.isdir(os.path.join(raw_seq_root_path, x))]
        for seq_name in tqdm(seq_names):
            frame_names = [x for x in os.listdir(os.path.join(raw_seq_root_path, seq_name)) if x.endswith('.ply')]
            frame_indices = [int(x.split('.')[1]) for x in frame_names]
            sort_indices = np.argsort(frame_indices)
            frame_names = [frame_names[x] for x in sort_indices]

            # create and save vertex sequence
            vert_seq = np.zeros((len(frame_names), 3, num_vertex))
            for i in tqdm(range(len(frame_names))):
                frame_mesh_path = os.path.join(raw_seq_root_path, seq_name, frame_names[i])
                frame_mesh = trimesh.load(frame_mesh_path, process=False)
                vert_seq[i] = frame_mesh.vertices.astype(np.float16).T      # [3, num_vertex], float16 for saving space
            np.savez_compressed(os.path.join(dst_video_path, seq_name + '.npz'), vert_seq=vert_seq)

            # copy audio
            shutil.copy(os.path.join(raw_audio_path, subj_name, seq_name + '.wav'),
                        os.path.join(dst_audio_path, seq_name + '.wav'))

            # render video
            frame_mesh = trimesh.load(os.path.join(raw_seq_root_path, seq_name, frame_names[0]), process=False)
            faces = frame_mesh.faces.astype(np.int32)
            render_video(os.path.join(dst_video_path, seq_name + '.mp4'),
                         os.path.join(dst_audio_path, seq_name + '.wav'),
                         vert_seq, faces, fps=fps, width=512)


if __name__ == '__main__':
    main()
