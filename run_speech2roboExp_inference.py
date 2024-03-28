import os
import argparse
import configparser
import trimesh
import numpy as np
import pickle
import torch
from speech2roboExp.model.audio_model import initialize_wav2vec2_model
from speech2roboExp.model.model import create_and_load_model
from speech2roboExp.inference import inference
from render_utils.render import render_video

# set global parameters
cfg_filepath = 'speech2roboExp/config/infer.cfg'

# load config
if not os.path.exists(cfg_filepath):
    raise FileNotFoundError('Config not found %s' % cfg_filepath)
cfg = configparser.RawConfigParser()
cfg.read(cfg_filepath)

# initialize
FPS = int(cfg['infer']['FPS'])                                # frame rate
SPEECH_WINDOW_SIZE = int(cfg['model']['SPEECH_WINDOW_SIZE'])  # sliding window size for speech features
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load template face mesh
template_mesh = trimesh.load(cfg['data']['TEMPLATE_MESH_FILEPATH'], process=False)
faces = template_mesh.faces.astype(np.int32)
neutral_vert = template_mesh.vertices.T

# initialize wav2vec2 model
print('Initializing wav2vec2 model...')
audio_processor, audio_model = initialize_wav2vec2_model()
audio_model.to(device).eval()

# initialize articulation model
print('Initializing articulation conv-net model...')
art_model, _ = create_and_load_model(cfg)
art_model.to(device).eval()

# initialize index mappings
subj_name2id_dict = pickle.load(open(cfg['data']['SUBJ_NAME2ID_FILEPATH'], 'rb'))
subj_one_hot_labels = np.eye(len(subj_name2id_dict.keys()))


def run_inference(audio_filepath: str,
                  subj: str,
                  result_path: str) -> None:
    """
    Run inference, input an audio file and output an animation video
    Args:
        audio_filepath: path to input audio file
        subj: subject name
        result_path: path to result directory
    Returns:
        None, (animation video is saved to result_path)
    """
    dst_filename = os.path.split(opt.audio_filepath)[-1].split('.')[0]

    subj_encoding = subj_one_hot_labels[:, subj_name2id_dict[subj]]

    # inference
    _, pred_vert_seq = inference(audio_filepath, subj_encoding, audio_processor, audio_model, art_model,
                                 SPEECH_WINDOW_SIZE, neutral_vert, device)

    # render animation
    render_video(os.path.join(result_path, dst_filename + '.mp4'), audio_filepath, pred_vert_seq, faces, FPS)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Infer articulation model')
    parser.add_argument('--audio_filepath', type=str,
                        help='path to input audio file', default='speech2roboExp/test_samples/jobs_speech_1.wav')
    parser.add_argument('--subj', type=str,
                        help='subject name', default='FaceTalk_170809_00138_TA')
    parser.add_argument('--result_path', type=str,
                        help='path to result directory', default='results/speech2roboExp')
    opt = parser.parse_args()

    # run inference
    run_inference(opt.audio_filepath, opt.subj, opt.result_path)
