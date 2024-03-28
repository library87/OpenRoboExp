import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from speech2roboExp.utils.utils import get_blendshape_names


class ArticulationModel(nn.Module):
    def __init__(self,
                 cfg: dict) -> None:
        """
        Articulation model
        Args:
            cfg: config file dictionary
        """
        super(ArticulationModel, self).__init__()
        speech_dim = int(cfg['model']['SPEECH_DIM'])
        speech_window_size = int(cfg['model']['SPEECH_WINDOW_SIZE'])
        subj_dim = len(pickle.load(open(cfg['data']['SUBJ_NAME2ID_FILEPATH'], 'rb')).keys())
        hid_dim = int(cfg['model']['HID_DIM'])
        dropout = float(cfg['model']['DROPOUT'])
        vert_dim = int(cfg['model']['VERT_DIM'])
        blendshape_dim = int(cfg['model']['BLENDSHAPE_DIM'])

        # load blendshape basis
        blendshapes_dict = pickle.load(open(cfg['data']['BLENDSHAPE_FILEPATH'], 'rb'))
        if len(blendshapes_dict.keys()) != blendshape_dim:
            raise ValueError('Number of blendshapes does not match!')
        blendshapes_names = get_blendshape_names()       # tuple, the order of blendshapes

        blendshape_basis = []
        for blendshape_name in blendshapes_names:
            blendshape_basis.append(np.asarray(blendshapes_dict[blendshape_name]))
        blendshape_basis = np.asarray(blendshape_basis)  # [blendshape_dim, num_vertex, 3]

        self.window_speech_encoder = WindowSpeechEncoder(speech_dim, speech_window_size, blendshape_dim, subj_dim,
                                                         hid_dim, dropout)
        self.vert_offset_decoder = VertexOffsetDecoder(blendshape_dim, vert_dim, blendshape_basis)

    def forward(self,
                window_speech_feat: torch.FloatTensor,
                subj_feat: torch.FloatTensor) -> torch.FloatTensor:
        """
        Infer vertex offset from windowed speech feature
        Args:
            window_speech_feat: [mini_batch_size, speech_window_size, speech_dim], mini_batch_size = seq_len
            subj_feat: [mini_batch_size, num_subject]
        Returns:
            vert_offset: [mini_batch_size, 3, num_vertex]
        """
        blendshapes = self.window_speech_encoder(window_speech_feat, subj_feat)  # [mini_batch_size, blendshape_dim]
        vert_offset = self.vert_offset_decoder(blendshapes)

        return vert_offset

    def infer_blendshape(self,
                         window_speech_feat: torch.FloatTensor,
                         subj_feat: torch.FloatTensor) -> torch.FloatTensor:
        """
        Infer blendshape coefficients from windowed speech feature
        Args:
            window_speech_feat: [mini_batch_size, speech_window_size, speech_dim], mini_batch_size = seq_len
            subj_feat: [mini_batch_size, num_subject]
        Returns:
            blendshapes: [mini_batch_size, blendshape_dim]
        """
        if window_speech_feat.dim() != 3 or subj_feat.dim() != 2:
            raise ValueError('Input data dimensions are not as expected.')
        blendshapes = self.window_speech_encoder(window_speech_feat, subj_feat)

        return blendshapes

    def decode_vert_offset(self,
                           blendshapes: torch.FloatTensor) -> torch.FloatTensor:
        """
        Decode vertex offset from blendshape coefficients
        Args:
            blendshapes: [mini_batch_size, blendshape_dim]
        Returns:
            vert_offset: [mini_batch_size, 3, num_vertex]
        """
        vert_offset = self.vert_offset_decoder(blendshapes)

        return vert_offset


def _create_res_block(hid_dim: int) -> nn.Module:
    """
    Create residual block
    Args:
        hid_dim: hidden dimension
    """
    return nn.Sequential(*[nn.Conv2d(hid_dim, hid_dim, (1, 3), (1, 2), (0, 1)),
                           nn.BatchNorm2d(hid_dim),
                           nn.ReLU(inplace=True),
                           nn.Conv2d(hid_dim, hid_dim, (1, 3), (1, 1), (0, 1)),
                           nn.BatchNorm2d(hid_dim)])


class StackedResNetBlock(nn.Module):
    def __init__(self,
                 speech_window_size: int,
                 hid_dim: int) -> None:
        """
        Stacked residual block
        Args:
            speech_window_size: sliding window size for speech features
            hid_dim: hidden dimension
        """
        super(StackedResNetBlock, self).__init__()
        self.num_res_blocks = int(np.log2(speech_window_size))
        self.blocks = nn.ModuleList()

        for _ in range(self.num_res_blocks):
            self.blocks.append(_create_res_block(hid_dim))

        self.downsample = nn.Sequential(*[nn.Conv2d(hid_dim, hid_dim, (1, 1), (1, 2), bias=False),
                                          nn.BatchNorm2d(hid_dim)])

    def forward(self,
                x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass
        Args:
            x: [mini_batch_size, hid_dim, 1, speech_window_size]
        Returns:
            x: [mini_batch_size, hid_dim]
        """
        residual = x
        for block in self.blocks:
            x = block(x)
            residual = self.downsample(residual)
            x += residual
            residual = x
        x = x.squeeze(dim=2).squeeze(dim=2)

        return x


class WindowSpeechEncoder(nn.Module):
    def __init__(self,
                 speech_dim: int,
                 speech_window_size: int,
                 blendshape_dim: int,
                 subj_dim: int,
                 hid_dim: int,
                 dropout: float = 0.1) -> None:
        """
        Windowed speech encoder model
        Args:
            speech_dim: speech feature dimension
            speech_window_size: sliding window size for speech features
            subj_dim: number of subjects
            hid_dim: hidden dimension
            dropout: dropout rate
        """
        super(WindowSpeechEncoder, self).__init__()
        self.dropout = dropout

        self.fc_speech = nn.Sequential(*[nn.Linear(speech_dim, hid_dim), nn.Dropout(self.dropout)])
        self.temporal_fusion = StackedResNetBlock(speech_window_size, hid_dim)

        self.fc_subj = nn.Sequential(*[nn.Linear(subj_dim, hid_dim), nn.Dropout(self.dropout)])

        self.fc1 = nn.Sequential(*[nn.Linear(hid_dim, 2 * hid_dim), nn.ReLU(), nn.Dropout(self.dropout)])
        self.fc2 = nn.Sequential(*[nn.Linear(2 * hid_dim, blendshape_dim), nn.Sigmoid()])  # normalize to [0, 1]

    def forward(self,
                window_speech_feat: torch.FloatTensor,
                subj_feat: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute blendshape coefficients from windowed speech feature
        Args:
            window_speech_feat: [mini_batch_size, speech_window_size, speech_dim], mini_batch_size = seq_len
            subj_feat: [mini_batch_size, num_subject]
        Returns:
            blendshapes: blendshape coefficients, [mini_batch_size, blendshape_dim]
        """
        if window_speech_feat.dim() != 3 or subj_feat.dim() != 2:
            raise ValueError('Input data dimensions are not as expected.')

        feat = self.fc_speech(window_speech_feat)
        feat = feat.permute(0, 2, 1)
        feat = torch.unsqueeze(feat, -1)
        feat = feat.permute(0, 1, 3, 2)  # [mini_batch_size, hid_dim, 1, speech_window_size]
        feat = self.temporal_fusion(feat)      # [mini_batch_size, hid_dim]

        subj_feat = self.fc_subj(subj_feat)
        feat = self.fc1(feat + subj_feat)
        blendshapes = self.fc2(feat)

        return blendshapes


class VertexOffsetDecoder(nn.Module):
    def __init__(self,
                 blendshape_dim: int,
                 vert_dim: int,
                 blendshape_basis: np.ndarray) -> None:
        """
        Vertex offset decoder model
        Args:
            blendshape_dim: blendshape dimension
            vert_dim: vertex dimension
            blendshape_basis: blendshape basis, [blendshape_dim, num_vertex, 3]
        """
        super(VertexOffsetDecoder, self).__init__()
        self.vert_dim = vert_dim

        self.fc = nn.Linear(blendshape_dim, 3 * self.vert_dim, bias=False)

        weight = torch.from_numpy(blendshape_basis).float()
        weight = weight.reshape((blendshape_dim, -1)).T     # [num_vertex * 3, blendshape_dim]
        self.fc.weight = nn.Parameter(weight)
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self,
                blendshapes: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute vertex offset from blendshape coefficients
        Args:
            blendshapes: [mini_batch_size, blendshape_dim]
        Returns:
            vert_offset: [mini_batch_size, 3, num_vertex]
        """
        vert_offset = self.fc(blendshapes)
        vert_offset = vert_offset.reshape((-1, self.vert_dim, 3)).permute(0, 2, 1)

        return vert_offset


def create_and_load_model(cfg: dict) -> (ArticulationModel, int):
    """
    Create and load model
    Args:
        cfg: config file dictionary
    Returns:
        model: model
        start_epoch: start epoch number
    """
    model = ArticulationModel(cfg)
    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    if bool(int(cfg['checkpoints']['FLAG_RESUME'])):
        checkpoint_filepath = os.path.join(cfg['checkpoints']['SAVE_DIR'], cfg['checkpoints']['RESUME_CHECKPOINT_NAME'])
        if os.path.isfile(checkpoint_filepath):
            checkpoint = torch.load(str(checkpoint_filepath))
            model.load_state_dict(checkpoint['state_dict'])
            if bool(int(cfg['checkpoints']['FLAG_RESET_EPOCH'])):
                start_epoch = 0
            else:
                start_epoch = checkpoint['epoch']
            print(f'=> loaded checkpoints {checkpoint_filepath} (epoch {checkpoint["epoch"]})')
        else:
            raise FileNotFoundError(f'=> no checkpoints found at {checkpoint_filepath}. '
                                    f'Please ensure the file exists and the path is correct.')
    else:
        start_epoch = 0

    return model, start_epoch
