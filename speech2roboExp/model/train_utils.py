from typing import Dict
from tabulate import tabulate
import torch


class AverageMeter(object):
    def __init__(self):
        """
        Computes and stores the average and current value
        """
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        """
        Reset all values
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update values in-place
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_losses() -> Dict[str, AverageMeter]:
    """
    Initialize losses
    Returns:
        losses_dict: dictionary of AverageMeter instances, containing the following:
            loss: AverageMeter
            pos_loss: AverageMeter
            face_pos_loss: AverageMeter
            mouth_pos_loss: AverageMeter
    """
    losses_dict = {
        'loss': AverageMeter(),
        'pos_loss': AverageMeter(),
        'face_pos_loss': AverageMeter(),
        'mouth_pos_loss': AverageMeter()
    }

    return losses_dict


def maintain_losses(losses_dict: Dict[str, AverageMeter],
                    loss_dict: Dict[str, torch.FloatTensor],
                    num_samples) -> None:
    """
    Maintain losses, update losses in-place
    Args:
        losses_dict: dictionary of AverageMeter instances, containing the following:
            loss: AverageMeter
            pos_loss: AverageMeter
            face_pos_loss: AverageMeter
            mouth_pos_loss: AverageMeter
        loss_dict: dictionary of losses for the current batch
            loss: overall loss
            pos_loss: position loss
            face_pos_loss: weighted face position loss
            mouth_pos_loss: weighted mouth position loss
        num_samples: number of samples
    """
    for loss_name, loss_value in loss_dict.items():
        losses_dict[loss_name].update(loss_value.item(), num_samples)


def print_losses(losses_dict: Dict[str, AverageMeter]) -> None:
    """
    Print losses
    Args:
        losses_dict: dictionary of AverageMeter instances, containing the following:
            loss: AverageMeter
            pos_loss: AverageMeter
            face_pos_loss: AverageMeter
            mouth_pos_loss: AverageMeter
    """
    tabulate_data = [['Loss', losses_dict['loss'].avg],
                     ['Pos Loss', losses_dict['pos_loss'].avg],
                     ['Face Pos Loss', losses_dict['face_pos_loss'].avg],
                     ['Mouth Pos Loss', losses_dict['mouth_pos_loss'].avg]]
    print('Count: ', losses_dict['losses'].count)
    print(tabulate(tabulate_data, headers=['Loss', 'Value'], tablefmt='fancy_grid'))


def epoch_time(time_elapsed: float) -> (int, int):
    """
    Convert time elapsed to minutes and seconds
    Args:
        time_elapsed: time elapsed in seconds
    Returns:
        elapsed_min: elapsed minutes
        elapsed_sec: elapsed seconds
    """
    elapsed_min = int(time_elapsed / 60)
    elapsed_sec = int(time_elapsed - (elapsed_min * 60))

    return elapsed_min, elapsed_sec


def print_train_val_logs(train_losses_dict: Dict[str, AverageMeter],
                         val_losses_dict: Dict[str, AverageMeter],
                         epoch: int,
                         time_elapsed: float) -> None:
    """
    Print train and validation logs
    Args:
        train_losses_dict: dictionary of training losses
        val_losses_dict: dictionary of validation losses
        epoch: epoch
        time_elapsed: time elapsed in seconds
    """
    elapsed_min, elapsed_sec = epoch_time(time_elapsed)
    tabulate_data = [['Loss', train_losses_dict['loss'].avg,
                      val_losses_dict['loss'].avg],
                     ['Pos Loss', train_losses_dict['pos_loss'].avg,
                      val_losses_dict['pos_loss'].avg],
                     ['Face Pos Loss', train_losses_dict['face_pos_loss'].avg,
                      val_losses_dict['face_pos_loss'].avg],
                     ['Mouth Pos Loss', train_losses_dict['mouth_pos_loss'].avg,
                      val_losses_dict['mouth_pos_loss'].avg]]

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {elapsed_min}m {elapsed_sec}s')
    print(tabulate(tabulate_data, headers=['Loss', 'Train', 'Val'], tablefmt='fancy_grid'))
