import numpy as np
import random


def create_dummy_sequence(length: int,
                          dimensions: int) -> np.ndarray:
    """
    Create a dummy sequence of zeros.
    Args:
        length: The length of the sequence.
        dimensions: The number of dimensions for each element in the sequence.
    Returns:
        A numpy array of zeros with the specified length and dimensions.
    """
    return np.zeros((length, dimensions))


def add_eye_blink_arkit(blendshape_seq: np.ndarray,
                        fps: int = 25) -> np.ndarray:
    """
    Add random eye blink to blendshape sequence in-place for ARKit blendshapes.
    Note: these are not learned eye blinks from real data, but just animated eye blinks
    Args:
        blendshape_seq: arkit blendshape sequence, [seq_len, blendshape_dim], blendshape_dim = 51
        fps: frame rate
    Returns:
        blendshape_seq: arkit blendshape sequence with eye blink, [seq_len, blendshape_dim]
    """
    EYE_BLINK_LEFT = 0
    EYE_BLINK_RIGHT = 7

    num_frames = blendshape_seq.shape[0]
    num_blinks = round(num_frames / fps / 3)  # average of 3 seconds to trigger a blink

    blink_weights = [[0, 0.35, 0.9, 1, 0.75, 0.3, 0.05, 0.15]]  # use list to allow for more complex blinks later

    freq = num_frames // (num_blinks + 1)
    weights = np.zeros(num_frames)
    for i in range(num_blinks):
        x1 = (i + 1) * freq + round(fps * (random.random() - 0.5))
        blink_duration = len(blink_weights[0])
        x2 = x1 + blink_duration
        if x1 >= 0 and x2 < weights.shape[0]:
            weights[x1:x2] = blink_weights[0] + np.random.uniform(-0.5, 0.5, blink_duration) * 0.05

    blendshape_seq[:, EYE_BLINK_LEFT] += weights  # eye-blink left
    blendshape_seq[:, EYE_BLINK_RIGHT] += weights  # eye-blink right

    return blendshape_seq
