import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import torch
from speech2roboExp.model.audio_model import initialize_wav2vec2_model, extract_speech_features


help_msg = """
Step 2 in prepare_dataset.
The preparation extracts pretrained audio features and save them in the same directory 
as the original audio file in .npy format.
The audio files should be organized as follows:
    <reorg_voca_path>
    ├── audio
    │   ├── <subj_1>
    │   │   ├── neutral
    │   │   │   ├── level_1
    │   │   │   │   ├── sentence01.wav
    │   │   │   │   ├── sentence02.wav
    │   ├── <subj_2>

Example usage:
python step2_extract_speech_features.py 
--audio_root_path /media/user/WD_BLACK/IROS24/Datasets/reorg/voca/audio
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_root_path', type=str, required=True,
                        help='Path to the audio folder.')
    opt = parser.parse_args()

    print(help_msg)

    # set parameters
    audio_root_path = opt.audio_root_path

    # initialize
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    audio_processor, audio_model = initialize_wav2vec2_model()
    audio_model.to(device).eval()

    # loop over the files
    audio_filepaths = glob.glob(audio_root_path + '/*/*/*/*.wav')
    for audio_filepath in tqdm(audio_filepaths):
        print('Processing...', audio_filepath)
        speech_seq = extract_speech_features(audio_filepath, audio_processor, audio_model, device)
        audio_path, audio_name = os.path.split(audio_filepath)
        np.save(str(os.path.join(audio_path, audio_name.split('.')[0] + '.npy')), speech_seq)
    print('Extraction done.')


if __name__ == "__main__":
    main()
