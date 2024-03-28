import numpy as np
import soundfile as sf
import resampy
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def initialize_wav2vec2_model() -> (Wav2Vec2Processor, Wav2Vec2ForCTC):
    """
    Initialize wav2vec2 model
    Returns:
        audio_processor: audio processor
        audio_model: audio model
    """
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    return audio_processor, audio_model


def extract_speech_features(audio_path: str,
                            audio_processor: Wav2Vec2Processor,
                            audio_model: Wav2Vec2ForCTC,
                            device: torch.device) -> np.ndarray:
    """
    Extract speech features from audio, speech features are extracted at 25 fps
    Args:
        audio_path: path to audio file
        audio_processor: wav2vec2 processor
        audio_model: wav2vec2 model
        device: device
    Returns:
        speech_seq: [seq_len, speech_dim], speech_dim = 392
    """
    # load audio and preprocess
    audio_data, sample_rate = sf.read(audio_path)
    if audio_data.ndim == 2 and audio_data.shape[1] == 2:
        audio_data = (audio_data[:, 0] + audio_data[:, 1]) / 2  # convert to mono
    if sample_rate != 16000:
        audio_data = resampy.resample(audio_data.astype(float), sample_rate, 16000)

    # extract features
    audio_tensor = audio_processor(audio_data, sample_rate=sample_rate, return_tensors='pt').input_values
    speech_seq = audio_model(audio_tensor.to(device)).logits.cpu().squeeze().detach().numpy()
    speech_seq = speech_seq[0::2, :]  # downsample, 50 -> 25 fps

    return speech_seq


def convert_to_window_feat(seq: np.ndarray,
                           window_size: int) -> np.ndarray:
    """
    Convert sequence to windowed features
    Args:
        seq: [seq_len, feat_dim]
        window_size: int
    Returns:
        window_feat: [seq_len, window_size, feat_dim]
    """
    if seq.ndim != 2:
        raise ValueError('Input data dimensions are not as expected.')
    seq_len = seq.shape[0]

    # add zero-padding at the beginning and end of the sequence
    seq = np.concatenate((np.zeros((window_size // 2, seq.shape[1])),
                          seq, np.zeros((window_size // 2, seq.shape[1]))), axis=0)

    # convert to window features
    window_feat = np.zeros((seq_len, window_size, seq.shape[1]))
    for j in range(0, seq_len):
        window_feat[j, :, :] = seq[j:j + window_size, :]

    return window_feat
