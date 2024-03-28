from typing import Dict
import numpy as np
import trimesh
import pickle
import torch
import torch.nn as nn
from torch.nn import functional


class PositionLoss(nn.Module):
    def __init__(self,
                 cfg: dict,
                 device: torch.device) -> None:
        """
        Compute position loss
        Args:
            cfg: config file dictionary
            device: device
        """
        super(PositionLoss, self).__init__()
        self.cfg = cfg
        self.device = device

        # load template face mesh
        template_mesh = trimesh.load(cfg['data']['TEMPLATE_MESH_FILEPATH'], process=False)
        self.faces = template_mesh.faces.astype(np.int32)                             # [num_faces, 3]
        self.neutral_vert = torch.FloatTensor(template_mesh.vertices.T).to(device)    # [3, num_vertex]

        # load vertex indices
        self.mouth_vert_indices = np.asarray(pickle.load(open(self.cfg['data']['MOUTH_VERT_INDICES_FILEPATH'], 'rb')))

        # load weights
        self.mouth_vert_weight = float(self.cfg['train']['MOUTH_VERT_WEIGHT'])

    def forward(self,
                pred_vert_offset_batch: torch.FloatTensor,
                gt_vert_offset_batch: torch.FloatTensor) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
        """
        Args:
            pred_vert_offset_batch: predicted vertex offset, [mini_batch_size, 3, num_vertex]
            gt_vert_offset_batch: ground truth vertex offset, [mini_batch_size, 3, num_vertex]
        Returns:
            loss: overall position loss
            face_loss: weighted face position loss
            mouth_loss: weighted mouth position loss
        """
        if pred_vert_offset_batch.dim() != 3 or gt_vert_offset_batch.dim() != 3:
            raise ValueError('Input data dimensions are not as expected.')

        face_loss = functional.mse_loss(pred_vert_offset_batch, gt_vert_offset_batch, reduction='sum')

        mouth_loss = functional.mse_loss(pred_vert_offset_batch[:, :, self.mouth_vert_indices],
                                         gt_vert_offset_batch[:, :, self.mouth_vert_indices], reduction='sum')
        mouth_loss = self.mouth_vert_weight * mouth_loss

        loss = face_loss + mouth_loss

        return loss, face_loss, mouth_loss


class OverallLoss(nn.Module):
    def __init__(self, cfg: dict, device: torch.device) -> None:
        """
        Compute overall loss
        Args:
            cfg: config file dictionary
            device: device
        """
        super(OverallLoss, self).__init__()
        self.cfg = cfg

        self.pos_loss = PositionLoss(cfg, device)

    def forward(self,
                pred_vert_offset_batch: torch.FloatTensor,
                gt_vert_offset_batch: torch.FloatTensor) -> Dict[str, torch.FloatTensor]:
        """
        Args:
            pred_vert_offset_batch: predicted vertex offset, [mini_batch_size, 3, num_vertex], mini_batch_size = seq_len
            gt_vert_offset_batch: ground truth vertex offset, [mini_batch_size, 3, num_vertex]
        Returns:
            loss_dict: dictionary of losses for the current batch
                loss: overall loss
                pos_loss: position loss
                face_pos_loss: face position loss
                mouth_pos_loss: weighted mouth position loss
        """
        if pred_vert_offset_batch.dim() != 3 or gt_vert_offset_batch.dim() != 3:
            raise ValueError('Input data dimensions are not as expected.')

        pos_loss, face_pos_loss, mouth_pos_loss = self.pos_loss(pred_vert_offset_batch, gt_vert_offset_batch)

        loss = pos_loss

        loss_dict = {
            'loss': loss,
            'pos_loss': pos_loss,
            'face_pos_loss': face_pos_loss,
            'mouth_pos_loss': mouth_pos_loss
        }

        return loss_dict
