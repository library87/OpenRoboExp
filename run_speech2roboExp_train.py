import os
import argparse
import configparser
from glob import glob
from time import time
import torch
from speech2roboExp.model.data_loader import create_dataloader
from speech2roboExp.model.model import create_and_load_model
from speech2roboExp.model.criterion import OverallLoss
from speech2roboExp.model.train_utils import print_train_val_logs
from speech2roboExp.train import train_one_epoch, evaluate_one_epoch


def run_train() -> None:
    # parse arguments
    parser = argparse.ArgumentParser(description='Train articulation model')
    parser.add_argument('--cfg_filepath',
                        type=str,
                        default='speech2roboExp/config/train.cfg',
                        help='path to config file')
    opt = parser.parse_args()

    # load config
    if not os.path.exists(opt.cfg_filepath):
        raise FileNotFoundError('Config not found %s' % opt.cfg_filepath)
    cfg = configparser.RawConfigParser()
    cfg.read(opt.cfg_filepath)

    # set parameters
    LEARNING_RATE = float(cfg['train']['LEARNING_RATE'])
    WEIGHT_DECAY = float(cfg['train']['WEIGHT_DECAY'])
    NUM_WORKERS = int(cfg['train']['NUM_WORKERS'])
    NUM_EPOCHS = int(cfg['train']['NUM_EPOCHS'])
    SAVE_INTERVAL = int(cfg['train']['SAVE_INTERVAL'])
    SPEECH_WINDOW_SIZE = int(cfg['model']['SPEECH_WINDOW_SIZE'])
    DATA_ROOT_PATH = cfg['data']['DATA_ROOT_PATH']
    SUBJ_NAME2ID_FILEPATH = cfg['data']['SUBJ_NAME2ID_FILEPATH']
    SAVE_DIR = cfg['checkpoints']['SAVE_DIR']

    # initialize
    train_filepaths = glob(os.path.join(DATA_ROOT_PATH, 'train', '*.pkl'))
    val_filepaths = glob(os.path.join(DATA_ROOT_PATH, 'val', '*.pkl'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(SAVE_DIR, exist_ok=True)

    # create dataloader
    train_loader = create_dataloader(train_filepaths,
                                     SUBJ_NAME2ID_FILEPATH,
                                     speech_window_size=SPEECH_WINDOW_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=True)
    val_loader = create_dataloader(val_filepaths,
                                   SUBJ_NAME2ID_FILEPATH,
                                   speech_window_size=SPEECH_WINDOW_SIZE,
                                   num_workers=NUM_WORKERS,
                                   shuffle=False)

    # create and load model
    model, start_epoch = create_and_load_model(cfg)
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # create criterion
    criterion = OverallLoss(cfg, device)

    # start training
    best_val_loss = float('inf')
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time()

        train_losses_dict = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_losses_dict = evaluate_one_epoch(model, val_loader, criterion, device)

        end_time = time()
        time_elapsed = end_time - start_time

        # save checkpoints
        if (epoch + 1) % SAVE_INTERVAL == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                       os.path.join(SAVE_DIR, 'epoch_%d.pth' % epoch))
        if val_losses_dict['loss'].avg < best_val_loss:
            best_val_loss = val_losses_dict['loss'].avg
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                       os.path.join(SAVE_DIR, 'best.pth'))

        torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                   os.path.join(SAVE_DIR, 'last.pth'))

        # print logs
        print_train_val_logs(train_losses_dict, val_losses_dict, epoch, time_elapsed)


if __name__ == '__main__':
    run_train()
