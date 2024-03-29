import os
from datetime import datetime
import time
import configparser
import gradio as gr
import torch.cuda
import trimesh
import numpy as np
import shutil
import pickle
from typing import List
from speech2roboExp.model.audio_model import initialize_wav2vec2_model
from speech2roboExp.model.model import create_and_load_model
from speech2roboExp.inference import inference, decode_vert_offset
from speech2roboExp.utils.utils import convert_offset_seq_to_vert_seq
from render_utils.render import render_video
from utils.utils import create_dummy_sequence, add_eye_blink_arkit
from utils.livelink_utils import write_to_livelink_face


def run_inference(audio_filepath: str,
                  subj: str = 'FaceTalk_170809_00138_TA',
                  eye_blink: str = 'Yes',
                  result_path: str = 'results/speech2roboExp/gui',
                  subj_mesh_path: str = 'speech2roboExp/assets/human_voca',
                  progress: gr.Progress = gr.Progress(track_tqdm=True)) -> (str, str, str,
                                                                            List[List[float]], List[List[float]]):
    """
    Run inference, input an audio file and output an animation video
    Args:
        audio_filepath: path to input audio file
        subj: subject name
        eye_blink: whether to add eye blink
        result_path: path to result directory
        subj_mesh_path: path to subject mesh directory
        progress: progress bar for gradio, track internal tqdm progress
    Returns:
        result_video_filepath: path to result video file
        result_livelink_filepath: path to result in livelink face format
        copied_audio_filepath: path to copied audio file
        time_cost: time cost, [[generation_time, render_time]], seconds
        speed: speed, [[generation_speed, render_speed]], frames per second
    """
    dst_filename = (f"{os.path.splitext(os.path.basename(audio_filepath))[0]}_{subj}_"
                    f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    start_time = time.time()

    # one-hot encodings
    subj_encoding = subj_one_hot_labels[:, subj_name2id_dict[subj]]

    # inference
    blendshape_seq, _ = inference(audio_filepath, subj_encoding,
                                  audio_processor, audio_model, art_model,
                                  SPEECH_WINDOW_SIZE, neutral_vert, device,
                                  flag_output_blendshape_only=True)

    # add eye blink optionally
    flag_add_eye_blink = eye_blink == 'Yes'
    if flag_add_eye_blink:
        blendshape_seq = add_eye_blink_arkit(blendshape_seq)

    end_time = time.time()
    generation_time = end_time - start_time
    print('Generation time: {:.2f} seconds'.format(generation_time))

    progress(0.9, desc='Decoding vertex sequence for rendering')
    # compute vertex offset
    vert_offset_seq = decode_vert_offset(blendshape_seq, art_model, device)

    # convert vertex offset to vertex
    subj_mesh = trimesh.load(os.path.join(subj_mesh_path, subj + '.obj'), process=False)
    subj_neutral_vert = subj_mesh.vertices.T
    vert_seq = convert_offset_seq_to_vert_seq(vert_offset_seq, subj_neutral_vert)

    # render animation
    result_video_filepath = os.path.join(result_path, dst_filename + '.mp4')
    render_video(result_video_filepath, audio_filepath, vert_seq, faces, fps=FPS)

    end_time = time.time()
    render_time = end_time - start_time - generation_time
    print('Render time: {:.2f} seconds'.format(render_time))

    # save blendshape sequence to livelink face format, TODO: replace dummy values
    result_livelink_filepath = os.path.join(result_path, dst_filename + '.txt')

    head_pos_seq = create_dummy_sequence(blendshape_seq.shape[0], 3)  # dummy head position
    head_rot_seq = create_dummy_sequence(blendshape_seq.shape[0], 3)  # dummy head rotation
    left_eye_rot_seq = create_dummy_sequence(blendshape_seq.shape[0], 2)  # dummy left eye rotation
    right_eye_rot_seq = create_dummy_sequence(blendshape_seq.shape[0], 2)  # dummy right eye rotation

    write_to_livelink_face(result_livelink_filepath, blendshape_seq, head_pos_seq,
                           head_rot_seq, left_eye_rot_seq, right_eye_rot_seq, fps=FPS)

    # copy audio file
    copied_audio_filepath = os.path.join(result_path, dst_filename + '.wav')
    shutil.copy(audio_filepath, copied_audio_filepath)

    return (result_video_filepath, result_livelink_filepath, copied_audio_filepath,
            [[generation_time, render_time]],
            [[int(blendshape_seq.shape[0] / generation_time), int(blendshape_seq.shape[0] / render_time)]])


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

# create gradio interface
demo = gr.Blocks(title="Speech2RoboExp Demo (v1.0)")
with demo:
    gr.Markdown(
        """
        ## Welcome to the Speech2RoboExp Experience (v1.0)!
        Get ready to bring your speech to life! Our cutting-edge Speech2RoboExp model can transform your voice into 
        dynamic facial expressions for both robot and 3D characters.
        
        Here's how it works:
        
        1. **Upload your speech audio**: Choose any audio file from your device.
        2. **Select a speaking style**: Customize the animation by selecting a speaking style that suits your character.
        3. **Generate**: Click the 'Generate' button and watch the magic happen!
        
        Note that the first run may take a bit longer due to the initial setup. Subsequent runs will be faster.
        You can also choose quick options from the examples provided below.
        
        In the output, you'll get a preview of the facial motion and a downloadable blendshape sequence in 
        ARKit standard. This sequence can be used to animate a 3D face model in 3D modeling software like Blender, 
        Maya, or Unreal Engine, giving you the freedom to create stunning visual narratives.
        
        Got questions or need help? Feel free to reach out to the author at [email](mailto:illiteratehit@gmail.com). 
        
        Dive in and enjoy the Speech2RoboExp experience!
        """
    )
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload"], type="filepath", label="Input speech", interactive=True)
            with gr.Row():
                subj_input = gr.Dropdown(label='Input speaking style', choices=list(subj_name2id_dict.keys()),
                                         value='FaceTalk_170809_00138_TA', interactive=True)
                eye_blink_input = gr.Radio(label='With eye blink', choices=['Yes', 'No'],
                                           value='Yes', interactive=True)
            generate_button = gr.Button(value="Generate", interactive=True)

            gr.Examples(
                [["speech2roboExp/test_samples/jobs_speech_1.wav", "FaceTalk_170809_00138_TA", "Yes"],
                 ["speech2roboExp/test_samples/kobe_retire_speech.wav", "FaceTalk_170725_00137_TA", "Yes"],
                 ["speech2roboExp/test_samples/churchill_speech_1.wav", "FaceTalk_170913_03279_TA", "Yes"],
                 ["speech2roboExp/test_samples/mawanglang.wav", "FaceTalk_170809_00138_TA", "Yes"],
                 ["speech2roboExp/test_samples/song_1.wav", "FaceTalk_170811_03275_TA", "Yes"]],
                inputs=[audio_input, subj_input, eye_blink_input],
                label="Examples"
            )

        with gr.Row():
            video_output = gr.Video(label="Generated facial motion preview", height=550, width=332,
                                    interactive=False)
            with gr.Column():
                gr.Markdown(
                    """
                    Output blendshape sequence
                    --------------------------
                    Use of ARKit blendshape standard.
                    """
                )
                livelink_file_output = gr.File(label="Livelink format", interactive=False)
                audio_file_output = gr.File(label="Paired audio", interactive=False)

                gr.Markdown(
                    """
                    Summary
                    -------
                    """
                )
                time_output = gr.Dataframe(headers=['Generation time', 'Render time'],
                                           datatype=['number', 'number'],
                                           interactive=False,
                                           label="Time cost [seconds]")
                speed_output = gr.Dataframe(headers=['Generation speed', 'Render speed'],
                                            datatype=['number', 'number'],
                                            interactive=False,
                                            label="Speed [frames per second]")

    generate_button.click(fn=run_inference, inputs=[audio_input, subj_input, eye_blink_input],
                          outputs=[video_output, livelink_file_output, audio_file_output, time_output, speed_output])

if __name__ == '__main__':
    # run gradio interface
    demo.queue()
    demo.launch()
    # demo.launch(server_name="0.0.0.0", server_port=7860)
