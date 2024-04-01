# VOCA Dataset Preparation

This document provides step-by-step instructions on how to prepare the VOCA dataset for the `speech2roboExp` project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Steps](#steps)
  - [Step 1: Organize Raw Data](#step-1-organize-raw-data)
  - [Step 2: Extract Speech Features](#step-2-extract-speech-features)
  - [Step 3: Convert Motion Data Frame Rate](#step-3-convert-motion-data-frame-rate)
  - [Step 4: Save Blendshapes](#step-4-save-blendshapes)
  - [Step 5: Create Dataset](#step-5-create-dataset)

## Prerequisites

Go to [VOCA](https://voca.is.tue.mpg.de/) and request to download the vocaset. Download the `unposedcleaneddata` and `audio` and unzip them. Organize the data as follows:

```bash
<raw_voca_path>
    ├── unposedcleaneddata
    │   ├── <subj_1>
    │   │   ├── sentence01
    │   │   │   ├── sentence01.000001.ply
    │   │   │   ├── sentence01.000002.ply
    │   │   ├── sentence02
    │   │   │   ├── sentence02.000001.ply
    │   │   │   ├── sentence02.000002.ply
    │   ├── <subj_2>
    ├── audio
    │   ├── <subj_1>
    │   │   ├── sentence01.wav
    │   │   ├── sentence02.wav
    │   ├── <subj_2>
```

## Steps

### Step 1: Organize Raw Data

Run the following command to organize the raw data:

```bash
python step1_organize_raw_data.py --raw_motion_path <raw_motion_path> --raw_audio_path <raw_audio_path> --dst_root_path <dst_root_path>
```
- `<raw_motion_path>`: path to the `unposedcleaneddata` folder
- `<raw_audio_path>`: path to the `audio` folder
- `<dst_root_path>`: path to the organized VOCA folder (60fps)
### Step 2: Extract Speech Features

Run the following command to extract the audio features and save them in the same directory as:

```bash
python step2_extract_speech_features.py --audio_root_path <audio_root_path>
```
- `<audio_root_path>`: path to the `audio` folder

### Step 3: Convert Motion Data Frame Rate

Run the following command to convert the frame rate of the motion data:

```bash
python step3_convert_motion_data_fps.py --src_root_path <src_root_path> --dst_root_path <dst_root_path> --template_mesh_filepath <template_mesh_filepath>
```
- `<src_root_path>`: path to the source VOCA data (60fps)
- `<dst_root_path>`: path to save the converted VOCA data (25fps)
- `<template_mesh_filepath>`: path to the template mesh (zero-shape zero-pose face mesh)
### Step 4: Save Blendshapes

Run the following command to save the blendshapes:

```bash
python step4_save_blendshapes.py --blendshape_path <blendshape_path> --dst_pkl_path <dst_pkl_path>
```
- `<blendshape_path>`: path to the zero-shape ARKit blendshape folder
- `<dst_pkl_path>`: path to save the pickle file containing blendshapes
### Step 5: Create Dataset

Run the following command to create the dataset:

```bash
python step5_create_dataset.py 
--data_root_path <data_root_path>
--subj_mesh_path <subj_mesh_path>
--out_root_path <out_root_path>
--num_val_seqs_per_subj <num_val_seqs_per_subj>
--num_test_seqs_per_subj <num_test_seqs_per_subj>
```
- `<data_root_path>`: path to the converted VOCA data (25fps)
- `<subj_mesh_path>`: path to the subject-specific zero-pose face mesh directory
- `<out_root_path>`: path to save the dataset
- `<num_val_seqs_per_subj>`: number of validation sequences per subject
- `<num_test_seqs_per_subj>`: number of test sequences per subject

Note: remember to copy the `subj_name2id.pkl` file from the <output_root_path> to `speech2roboExp/assets/human_voca` folder after running the script.
