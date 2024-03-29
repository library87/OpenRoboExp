from typing import Tuple
import numpy as np
import csv


def get_livelink_ordered_blendshape_names() -> Tuple:
    """
    Get ordered blendshape names for livelink face,
    the order adapts to our own blendshape order, the names follow livelink blendshape names
    Returns:
        ordered_blendshape_names: Tuple of ordered blendshape names
    """
    ordered_blendshape_names = ('eyeBlink_L', 'eyeLookDown_L', 'eyeLookIn_L', 'eyeLookOut_L',
                                'eyeLookUp_L', 'eyeSquint_L', 'eyeWide_L', 'eyeBlink_R',
                                'eyeLookDown_R', 'eyeLookIn_R', 'eyeLookOut_R', 'eyeLookUp_R',
                                'eyeSquint_R', 'eyeWide_R', 'jawForward', 'jawLeft', 'jawRight',
                                'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
                                'mouthSmile_L', 'mouthFrown_L', 'mouthDimple_L', 'mouthStretch_L',
                                'mouthRight', 'mouthSmile_R', 'mouthFrown_R', 'mouthDimple_R',
                                'mouthStretch_R', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
                                'mouthShrugUpper', 'mouthPress_L', 'mouthPress_R', 'mouthLowerDown_L',
                                'mouthLowerDown_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'browDown_L',
                                'browDown_R', 'browInnerUp', 'browOuterUp_L', 'browOuterUp_R',
                                'cheekPuff', 'cheekSquint_L', 'cheekSquint_R', 'noseSneer_L',
                                'noseSneer_R')  # Note: the order of blendshapes is important

    return ordered_blendshape_names


def write_to_livelink_face(dst_filepath: str,
                           blendshape_seq: np.ndarray,
                           head_pos_seq: np.ndarray,
                           head_rot_seq: np.ndarray,
                           left_eye_rot_seq: np.ndarray,
                           right_eye_rot_seq: np.ndarray,
                           fps: int = 25) -> None:
    """
    Write face motion to a file in livelink face format
    Args:
        dst_filepath: path to destination file, csv format but with .txt extension for livelink
        blendshape_seq: [num_frames, blendshape_dim], face motion
        head_pos_seq: [num_frames, 3], head position, in meters
        head_rot_seq: [num_frames, 3], head rotation, in degrees
        left_eye_rot_seq: [num_frames, 2], left eye rotation, in degrees
        right_eye_rot_seq: [num_frames, 2], right eye rotation, in degrees
        fps: frame rate
    """
    timestamp = np.arange(blendshape_seq.shape[0]) / fps

    with open(dst_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # write header, livelink blendshape order
        csv_blendshape_names = ['browInnerUp', 'browDown_L', 'browDown_R', 'browOuterUp_L',
                                'browOuterUp_R', 'eyeLookUp_L', 'eyeLookUp_R', 'eyeLookDown_L',
                                'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L',
                                'eyeLookOut_R', 'eyeBlink_L', 'eyeBlink_R', 'eyeSquint_L', 'eyeSquint_R',
                                'eyeWide_L', 'eyeWide_R', 'cheekPuff', 'cheekSquint_L', 'cheekSquint_R', 'noseSneer_L',
                                'noseSneer_R', 'jawOpen', 'jawForward', 'jawLeft', 'jawRight', 'mouthFunnel',
                                'mouthPucker', 'mouthLeft', 'mouthRight', 'mouthRollUpper', 'mouthRollLower',
                                'mouthShrugUpper', 'mouthShrugLower', 'mouthClose', 'mouthSmile_L', 'mouthSmile_R',
                                'mouthFrown_L', 'mouthFrown_R', 'mouthDimple_L', 'mouthDimple_R', 'mouthUpperUp_L',
                                'mouthUpperUp_R', 'mouthLowerDown_L', 'mouthLowerDown_R', 'mouthPress_L',
                                'mouthPress_R', 'mouthStretch_L', 'mouthStretch_R', 'tongueOut']

        writer.writerow(['bs'] + csv_blendshape_names)

        # create blendshape dictionary
        ordered_blendshape_names = get_livelink_ordered_blendshape_names()
        blendshape_dict = {blendshape_name: blendshape_seq[:, i] for i, blendshape_name in
                           enumerate(ordered_blendshape_names)}
        blendshape_dict['tongueOut'] = np.zeros(timestamp.shape[0], dtype=np.float32)  # dummy tongueOut

        # write data
        for i in range(timestamp.shape[0]):
            writer.writerow(['k', int(timestamp[i] * 1000)] +
                            [head_pos_seq[i, 0], head_pos_seq[i, 1], head_pos_seq[i, 2],
                             head_rot_seq[i, 0], head_rot_seq[i, 1], head_rot_seq[i, 2],
                             left_eye_rot_seq[i, 0], left_eye_rot_seq[i, 1],
                             right_eye_rot_seq[i, 0], right_eye_rot_seq[i, 1]] +
                            [blendshape_dict[blendshape_name][i] for blendshape_name in csv_blendshape_names])
