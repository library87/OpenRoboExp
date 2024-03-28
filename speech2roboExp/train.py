from typing import Dict
from tqdm import tqdm
import torch
from torch.utils import data
from speech2roboExp.model.train_utils import initialize_losses, maintain_losses, print_losses
from speech2roboExp.model.model import ArticulationModel
from speech2roboExp.model.criterion import OverallLoss
from speech2roboExp.model.train_utils import AverageMeter


def train_one_epoch(model: ArticulationModel,
                    train_loader: data.DataLoader,
                    optimizer: torch.optim,
                    criterion: OverallLoss,
                    device: torch.device,
                    log_interval: int = 500) -> Dict[str, AverageMeter]:
    """
    Train one epoch
    Args:
        model: articulation conv-net model
        train_loader: training data loader
        optimizer: optimizer
        criterion: loss function
        device: device
        log_interval: log interval
    Returns:
        dictionary of AverageMeter instances, containing the following:
            loss: AverageMeter
            pos_loss: AverageMeter
            face_pos_loss: AverageMeter
            mouth_pos_loss: AverageMeter
    """
    model.train()

    losses_dict = initialize_losses()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (window_speech_seq, gt_vert_offset_seq, subj_seq) in pbar:
        optimizer.zero_grad()

        window_speech_seq = window_speech_seq.to(device)
        gt_vert_offset_seq = gt_vert_offset_seq.to(device)
        subj_seq = subj_seq.to(device)

        pred_vert_offset_seq = model(window_speech_seq, subj_seq)

        loss_dict = criterion(pred_vert_offset_seq, gt_vert_offset_seq)
        loss_dict['loss'].backward()  # backprop using overall loss

        optimizer.step()

        # maintain losses
        num_samples = gt_vert_offset_seq.shape[0]
        maintain_losses(losses_dict, loss_dict, num_samples)

        # print losses
        if i > 0 and i % log_interval == 0:
            print_losses(losses_dict)

    return losses_dict


def evaluate_one_epoch(model: ArticulationModel,
                       val_loader: data.DataLoader,
                       criterion: OverallLoss,
                       device: torch.device) -> Dict[str, AverageMeter]:
    """
    Evaluate one epoch
    Args:
        model: articulation conv-net model
        val_loader: validation data loader
        criterion: loss function
        device: device
    Returns:
        dictionary of AverageMeter instances, containing the following:
            loss: AverageMeter
            pos_loss: AverageMeter
            face_pos_loss: AverageMeter
            mouth_pos_loss: AverageMeter
    """
    model.eval()

    losses_dict = initialize_losses()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (window_speech_seq, gt_vert_offset_seq, subj_seq) in pbar:
        window_speech_seq = window_speech_seq.to(device)
        gt_vert_offset_seq = gt_vert_offset_seq.to(device)
        subj_seq = subj_seq.to(device)

        pred_vert_offset_seq = model(window_speech_seq, subj_seq)
        loss_lst = criterion(pred_vert_offset_seq, gt_vert_offset_seq)

        # maintain losses
        num_samples = gt_vert_offset_seq.shape[0]
        maintain_losses(losses_dict, loss_lst, num_samples)

    return losses_dict
