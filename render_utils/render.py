from typing import Tuple
import os
import numpy as np
import subprocess
import imageio
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from render_utils import Sim3DR_Cython


def norm_vecs(arr: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length
    Args:
        arr: [num_vecs, 3]
    Returns:
        arr: [num_vecs, 3], with unit length
    """

    return arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]


def get_vert_normal(vert: np.ndarray,
                    faces: np.ndarray) -> np.ndarray:
    """
    Get vertex normal
    Args:
        vert: [num_vertex, 3]
        faces: [num_faces, 3]
    Returns:
        normal: [num_vertex, 3]
    """
    vert_normal = np.zeros_like(vert, dtype=np.float32)
    Sim3DR_Cython.get_normal(vert_normal, vert, faces, vert.shape[0], faces.shape[0])

    return vert_normal


def rasterize(vert: np.ndarray,
              faces: np.ndarray,
              colors: np.ndarray,
              img: np.ndarray) -> np.ndarray:
    """
    Rasterize mesh to image
    Args:
        vert: [num_vertex, 3]
        faces: [num_faces, 3]
        colors: [num_vertex, 3]
        img: [height, width, channel]
    Returns:
        img: [height, width, channel]
    """
    height, width, channel = img.shape

    depth_buffer = np.zeros((height, width), dtype=np.float32) - 1e8
    Sim3DR_Cython.rasterize(img, vert, faces, colors, depth_buffer, faces.shape[0],
                            height, width, channel, reverse=True)

    return img


def norm_vert(vert: np.ndarray) -> np.ndarray:
    """
    Normalize vertices to [-1, 1]
    Args:
        vert: [num_vertex, 3]
    Returns:
        vert: [num_vertex, 3]
    """
    vert -= vert.min(0)[None, :]
    vert /= vert.max()
    vert *= 2
    vert -= vert.max(0)[None, :] / 2

    return vert


class RenderPipeline(object):
    def __init__(self):
        self.intensity_ambient = np.array([0.0], dtype=np.float32)
        self.color_ambient = np.array([[255, 255, 255]], dtype=np.float32) / 255
        self.intensity_directional = np.array([0.5, 0.5, 1, 0.25], dtype=np.float32) * 0.95
        self.color_directional = np.array([[169, 169, 169]], dtype=np.float32) / 255
        self.light_pos = np.array([[2.5, 0, 2.5],
                                   [-2.5, 0, 2.5],
                                   [0, 2.5, 2.5],
                                   [0, -2.5, 2.5]], dtype=np.float32)

    def __call__(self, vert: np.ndarray, faces: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Compute the color of each pixel in the image
        Args:
            vert: [num_vertex, 3]
            faces: [num_faces, 3]
            img: [height, width, channel]
        Returns:
            img: [height, width, channel]
        """
        normal = get_vert_normal(vert, faces)

        # ambient component
        scene_rad = np.zeros_like(vert, dtype=np.float32)
        scene_rad += self.intensity_ambient * self.color_ambient

        # diffuse component
        for i in range(self.light_pos.shape[0]):
            light_dir = norm_vecs(self.light_pos[i] - norm_vert(vert.copy()))
            surf_irrad = np.clip(np.sum(normal * light_dir, axis=1)[:, None], 0, 1)
            scene_rad += self.intensity_directional[i] * self.color_directional * surf_irrad

        scene_rad = np.clip(scene_rad, 0, 1)

        # rasterize mesh to image
        img = rasterize(vert, faces, scene_rad, img)

        return img


def render_img(vert: np.ndarray,
               faces: np.ndarray,
               aspect_ratio: np.ndarray,
               render_pipeline: RenderPipeline,
               width: int = 512) -> np.ndarray:
    """
    Render mesh to image
    Args:
        vert: [3, num_vertex]
        faces: [num_faces, 3]
        aspect_ratio: [3]
        render_pipeline: RenderPipeline
        width: int, render image width
    Returns:
        img: [height, width, channel]
    """
    def _to_c_contiguous_type(arr: np.ndarray) -> np.ndarray:
        if not arr.flags.c_contiguous:
            return arr.copy(order='C')

        return arr

    for i in range(0, 3):
        vert[i, :] = vert[i, :] / aspect_ratio[1] * aspect_ratio[i] * width

    img = np.zeros((width, int(width / aspect_ratio[1] * aspect_ratio[0]), 3), dtype=np.uint8)
    img = render_pipeline(_to_c_contiguous_type(vert.T),
                          _to_c_contiguous_type(faces), img)  # transpose vert to [num_vertex, 3]

    return img


def norm_vert_seq(vert_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize vertex sequence to [-1, 1] and calculate aspect ratio
    Args:
        vert_seq: [seq_len, 3, num_vertex]
    Returns:
        vert_seq: [seq_len, 3, num_vertex]
        aspect_ratio: [3]
    """
    max_vals = np.asarray([np.max(vert_seq[:, 0, :]), np.max(vert_seq[:, 1, :]), np.max(vert_seq[:, 2, :])])
    min_vals = np.asarray([np.min(vert_seq[:, 0, :]), np.min(vert_seq[:, 1, :]), np.min(vert_seq[:, 2, :])])
    aspect_ratio = max_vals - min_vals

    # Check for division by zero
    if np.any(max_vals == min_vals):
        raise ValueError("Max and min values for a dimension are the same, leading to division by zero.")

    max_vals = np.tile(np.expand_dims(np.tile(np.expand_dims(max_vals, axis=0), (vert_seq.shape[0], 1)), axis=2),
                       (1, 1, vert_seq.shape[2]))
    min_vals = np.tile(np.expand_dims(np.tile(np.expand_dims(min_vals, axis=0), (vert_seq.shape[0], 1)), axis=2),
                       (1, 1, vert_seq.shape[2]))

    vert_seq = (vert_seq - min_vals) / (max_vals - min_vals)

    return vert_seq, aspect_ratio


def rotate_vert_seq(vert_seq: np.ndarray,
                    rot_y: float = -13,
                    rot_x: float = 5) -> np.ndarray:
    """
    Rotate vertex sequence for visualization
    Args:
        vert_seq: [seq_len, 3, num_vertex]
        rot_y: float, rotation angle around y-axis in degrees
        rot_x: float, rotation angle around x-axis in degrees
    Returns:
        vert_seq: [seq_len, 3, num_vertex]
    """
    R1 = Rotation.from_euler('y', rot_y, degrees=True)
    R2 = Rotation.from_euler('x', rot_x, degrees=True)
    for i in range(vert_seq.shape[0]):
        vert_seq[i] = np.matmul(R1.as_matrix(), vert_seq[i])
        vert_seq[i] = np.matmul(R2.as_matrix(), vert_seq[i])

    return vert_seq


def render_video(dst_video_filepath: str,
                 audio_filepath: str,
                 vert_seq: np.ndarray,
                 faces: np.ndarray,
                 fps: int = 25,
                 width: int = 512) -> None:
    """
    Render video from vertex sequence
    Args:
        dst_video_filepath: str
        audio_filepath: str
        vert_seq: [seq_len, 3, num_vertex]
        faces: [num_faces, 3]
        fps: int, video frame rate
        width: int, render image width
    Returns:
        None
    """
    print('Rendering video...')
    render_pipeline = RenderPipeline()

    vert_seq = vert_seq.astype(np.float32)

    # rotate vertex sequence for visualization
    vert_seq = rotate_vert_seq(vert_seq)

    # normalize vertex sequence to [-1, 1] to adapt render pipeline
    vert_seq, aspect_ratio = norm_vert_seq(vert_seq)

    # render
    dst_path = os.path.split(dst_video_filepath)[0]
    os.makedirs(dst_path, exist_ok=True)

    tmp_filepath = os.path.join(dst_path, 'tmp.mp4')
    writer = imageio.get_writer(tmp_filepath, fps=fps)
    for i in tqdm(range(vert_seq.shape[0])):
        vert = vert_seq[i, :, :]
        img = render_img(vert.copy(), faces.copy(), aspect_ratio, render_pipeline, width)
        writer.append_data(img)
    writer.close()

    subprocess.run(['ffmpeg', '-y', '-i', tmp_filepath, '-i', audio_filepath, '-c:v', 'copy', '-c:a', 'aac',
                    dst_video_filepath])
    os.remove(tmp_filepath)
    print('Rendering successful!')
