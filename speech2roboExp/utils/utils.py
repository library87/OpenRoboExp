from typing import Tuple
import numpy as np
from scipy.signal import butter, lfilter


def butter_lowpass_filter(data: np.ndarray,
                          cutoff: float,
                          fps: int,
                          order: int) -> np.ndarray:
    """
    Butterworth lowpass filter
    Args:
        data: [seq_len, num_feats]
        cutoff: cutoff frequency
        fps: frame rate
        order: filter order
    Returns:
        data: [seq_len, num_feats]
    """
    b, a = butter(order, cutoff, btype='low', fs=fps)
    data = lfilter(b, a, data.copy(), axis=0)

    return data


def convert_offset_seq_to_vert_seq(vert_offset_seq: np.ndarray,
                                   neutral_vert: np.ndarray) -> np.ndarray:
    """
    Convert vertex offset sequence to vertex sequence
    Args:
        vert_offset_seq: [seq_len, 3, num_vertex]
        neutral_vert: zero-pose vertex, [3, num_vertex]
    Returns:
        vert_seq: [seq_len, 3, num_vertex]
    """
    vert_seq = np.zeros((vert_offset_seq.shape[0], 3, vert_offset_seq.shape[2]))
    for i in range(vert_offset_seq.shape[0]):
        vert_seq[i, :, :] = vert_offset_seq[i, :, :] + neutral_vert

    return vert_seq


def get_blendshape_names() -> Tuple:
    """
    Get blendshape names
    Returns:
        blendshape_names: tuple of blendshape names
    """
    blendshape_names = ('eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft',
                        'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight',
                        'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
                        'eyeSquintRight', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawRight',
                        'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
                        'mouthSmileLeft', 'mouthFrownLeft', 'mouthDimpleLeft', 'mouthStretchLeft',
                        'mouthRight', 'mouthSmileRight', 'mouthFrownRight', 'mouthDimpleRight',
                        'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
                        'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft',
                        'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'browDownLeft',
                        'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
                        'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft',
                        'noseSneerRight')  # Note: the order of blendshapes is important

    return blendshape_names
