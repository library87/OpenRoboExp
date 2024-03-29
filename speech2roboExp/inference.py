import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speech2roboExp.model.model import ArticulationModel
from speech2roboExp.model.audio_model import extract_speech_features, convert_to_window_feat
from speech2roboExp.utils.utils import butter_lowpass_filter, convert_offset_seq_to_vert_seq


def infer_blendshape(audio_filepath: str,
                     subj_encoding: np.ndarray,
                     audio_processor: Wav2Vec2Processor,
                     audio_model: Wav2Vec2ForCTC,
                     art_model: ArticulationModel,
                     speech_window_size: int,
                     device: torch.device) -> np.ndarray:
    """
    Infer blendshape coefficients from audio file
    Args:
        audio_filepath: path to audio file
        subj_encoding: subject one-hot encoding, [num_subject]
        audio_processor: wav2vec2 processor
        audio_model: wav2vec2 model
        art_model: articulation model
        speech_window_size: sliding window size for speech features
        device: device
    Returns:
        blendshape_seq: arkit blendshape coefficient sequence, [seq_len, blendshape_dim]
    """
    # extract speech features
    speech_seq = extract_speech_features(audio_filepath, audio_processor, audio_model, device)

    # convert to window features
    window_speech_seq = convert_to_window_feat(speech_seq, speech_window_size)

    # infer blendshape coefficients
    subj_seq = np.tile(subj_encoding, (window_speech_seq.shape[0], 1))
    blendshape_seq = art_model.infer_blendshape(torch.FloatTensor(window_speech_seq).to(device),
                                                torch.FloatTensor(subj_seq).to(device))

    return blendshape_seq.cpu().squeeze().detach().numpy()


def filter_blendshape(blendshape_seq: np.ndarray,
                      fps: int = 25) -> np.ndarray:
    """
    Filter blendshape coefficients
    Args:
        blendshape_seq: blendshape coefficient sequence, [seq_len, blendshape_dim]
        fps: frame rate
    Returns:
        filtered_blendshape_seq: filtered arkit blendshape coefficient sequence, [seq_len, blendshape_dim]
    """
    filtered_blendshape_seq = butter_lowpass_filter(blendshape_seq, cutoff=6.5, fps=fps, order=5)

    return filtered_blendshape_seq


def decode_vert_offset(blendshape_seq: np.ndarray,
                       art_model: ArticulationModel,
                       device: torch.device) -> np.ndarray:
    """
    Decode vertex offset from blendshape coefficients
    Args:
        blendshape_seq: arkit blendshape coefficient sequence, [seq_len, blendshape_dim], blendshape_dim = 51
        art_model: articulation model
        device: device
    Returns:
        vert_offset_seq: vertex offset sequence, [seq_len, 3, num_vertex]
    """
    vert_offset_seq = art_model.decode_vert_offset(torch.FloatTensor(blendshape_seq).to(device))
    vert_offset_seq = vert_offset_seq.cpu().squeeze().detach().numpy()

    return vert_offset_seq


def inference(audio_filepath: str,
              subj_encoding: np.ndarray,
              audio_processor: Wav2Vec2Processor,
              audio_model: Wav2Vec2ForCTC,
              art_model: ArticulationModel,
              speech_window_size: int,
              neutral_vert: np.ndarray,
              device: torch.device,
              flag_output_blendshape_only: bool = False) -> (np.ndarray, np.ndarray):
    """
    Run inference, input an audio file and output a sequence of vertices
    Args:
        audio_filepath: path to audio file
        subj_encoding: subject one-hot encoding, [num_subject]
        audio_processor: wav2vec2 processor
        audio_model: wav2vec2 model
        art_model: articulation model
        speech_window_size: sliding window size for speech features
        neutral_vert: neutral vertex, [3, num_vertex]
        device: device
        flag_output_blendshape_only: flag to output blendshape only
    Returns:
        blendshape_seq: arkit blendshape coefficient sequence, [seq_len, blendshape_dim]
        vert_seq: vertex sequence, [seq_len, 3, num_vertex]
    """
    # infer blendshape coefficients
    blendshape_seq = infer_blendshape(audio_filepath, subj_encoding, audio_processor, audio_model,
                                      art_model, speech_window_size, device)

    # filter blendshape coefficients
    blendshape_seq = filter_blendshape(blendshape_seq)

    if flag_output_blendshape_only:
        return blendshape_seq, None

    # compute vertex offset
    vert_offset_seq = decode_vert_offset(blendshape_seq, art_model, device)

    # convert vertex offset to vertex
    vert_seq = convert_offset_seq_to_vert_seq(vert_offset_seq, neutral_vert)

    return blendshape_seq, vert_seq
