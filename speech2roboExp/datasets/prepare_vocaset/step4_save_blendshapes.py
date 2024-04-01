import os
import copy
import trimesh
import pickle
from tqdm import tqdm
import argparse
from speech2roboExp.utils.utils import get_blendshape_names

help_msg = """
Step 4 in prepare_dataset. Save blendshapes from .obj to pickle file. 

Example usage:
python step4_save_blendshapes.py
--blendshape_path speech2roboExp/datasets/prepare_vocaset/human_voca_blendshapes
--dst_pkl_path speech2roboExp/assets/human_voca/template_arkit_blendshapes.pkl
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--blendshape_path', type=str, required=True,
                        help='The path to the zero-shape ARKit blendshape folder.')
    parser.add_argument('--dst_pkl_path', type=str, required=True,
                        help='The path to save the pickle file containing blendshapes.')
    opt = parser.parse_args()

    print(help_msg)

    # set parameters
    blendshape_path = opt.blendshape_path
    dst_pkl_path = opt.dst_pkl_path

    # get blendshape names
    blendshape_names = get_blendshape_names()   # Note: the order of blendshapes is important, ARKit blendshapes

    # read the blendshapes
    template_mesh = trimesh.load(os.path.join(blendshape_path, 'ABase.obj'), process=False)
    blendshapes = dict()
    for i, blendshape_name in tqdm(enumerate(blendshape_names)):
        model_mesh = trimesh.load(os.path.join(blendshape_path, blendshape_name + '.obj'), process=False)
        blendshapes[blendshape_name] = copy.deepcopy(model_mesh.vertices - template_mesh.vertices)

    # save to pickle
    pickle.dump(blendshapes, open(dst_pkl_path, 'wb'))


if __name__ == "__main__":
    main()
