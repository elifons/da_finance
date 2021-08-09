from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os.path as op
import numpy as np
from utils.models import LSTMNet
from utils.pytorchtools import EarlyStopping
from utils.mySummary import SummaryLogger
import torch
import os
import torch.nn as nn
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.train_utils import evaluate_model, epoch_trainer, epoch_validation
import argparse
import time
import yaml
import shutil
import tsaug as ts
from utils.augmentation import * 

da_methods_mapping = {'convolve': ts.Convolve(window="hann"),
                    'pool': ts.Pool(size=3),
                    'jitter': ts.AddNoise(scale=0.05),
                    'quantize': ts.Quantize(n_levels=17),
                    'reverse': ts.Reverse(),
                    'timewarp': ts.TimeWarp(n_speed_change=4, max_speed_ratio=1.5),
                    'spawner': spawner,
                    'scaling': scaling,
                    'magnitude_warp': magnitude_warp,
                    'window_warp': window_warp
                    }


# data_dir = op.join(op.expanduser('~'), 'data/')

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu_number)
    print(torch.cuda.current_device())
else:
    device = torch.device("cpu")
print(device)

def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass



def build_dataloader(x_data, y_data, batch_size, shuffle=True):
        train_data = TensorDataset(torch.from_numpy(x_data).float(), torch.from_numpy(y_data))
        train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size, drop_last=False)
        return train_loader


def augment_dataset(i_sp, batch_size, da_method, augment_times=1):
    data_dir = 'data'
    train_x = np.load(op.join(data_dir, 'study_period_X_'+str(i_sp)+'_train.npy'))
    train_y = np.load(op.join(data_dir, 'study_period_Y_'+str(i_sp)+'_train.npy'))

    validation_split = 0.2
    dataset_size=train_x.shape[0]
    indices = list(range(dataset_size))
    split = dataset_size - int(np.floor(validation_split*dataset_size))

    trainX, trainY = train_x[:split], train_y[:split]
    if da_method in ['convolve', 'pool', 'jitter', 'quantize', 'reverse', 'timewarp']:
        trainX = np.concatenate([trainX, *[da_methods_mapping[da_method].augment(trainX) for i in range(augment_times)]])
        trainY = np.concatenate([trainY, *[trainY for i in range(augment_times)]])
    elif da_method in ['magnitude_warp', 'window_warp', 'scaling']:
        trainX = np.concatenate([trainX, *[da_methods_mapping[da_method](trainX) for i in range(augment_times)]])
        trainY = np.concatenate([trainY, *[trainY for i in range(augment_times)]])
    train_loader = build_dataloader(trainX, trainY, batch_size=batch_size)
    valid_loader = build_dataloader(train_x[split:], train_y[split:], batch_size=batch_size)
    return train_loader, valid_loader



def train_eval_single_model(model, train_loader, valid_loader, n_epochs, path, i_sp, device, patience):
    logger = SummaryLogger(path)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
    print('Start training')
    for epoch in range(n_epochs):
        counter = 0
        loss, acc = epoch_trainer(model, train_loader, optimizer, criterion, logger, device)
        valid_loss, valid_acc = epoch_validation(model, valid_loader, logger, device)
        print(epoch, loss, acc, valid_loss, valid_acc)
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break       
    logger.close()
    model_file_name = os.path.join(path, 'checkpoint.pt')
    model.load_state_dict(torch.load(model_file_name))
    metrics = evaluate_model(model, path, i_sp, device)
    return metrics


def run_experiments(args):
    run_path = args.run_path
    create_directory(op.join(run_path, 'output'))
    for i in range(args.init_sp, args.end_sp):
        path = op.join(run_path, 'output/study_period_'+str(i).zfill(2))
        create_directory(path)
        train_loader, valid_loader = augment_dataset(i, batch_size=args.batch_size, da_method=args.da_method)
        model = LSTMNet(1, hidden_dim=args.hidden_dim, output_dim=2, n_layers=args.n_layers, device=device)
        model.to(device)
        metrics = train_eval_single_model(model, train_loader, valid_loader, args.n_epochs, path, i, device, args.patience)
        print(metrics)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run_path', default='./', help='experiment directoy')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--hidden_dim', type=int, default=25, help='hidden dimension of LSTM')
parser.add_argument('--n_layers', type=int, default=1, help='number of layers in the LSTM')
parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs for training')
parser.add_argument('--init_sp', type=int, default=0, help='initial data split')
parser.add_argument('--end_sp', type=int, default=1, help='final data split')
# parser.add_argument('--gpu_number', type=int, default=0, help=' ')
parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
parser.add_argument('--da_method', choices=list(da_methods_mapping.keys())+['None'], default='None', help='augmentation methods')
args = parser.parse_args()
 
if __name__ == "__main__":
    copied_script_name = op.basename(__file__)
    if (args.run_path != './'):
        os.popen('./cpfiles.sh '+args.run_path).read()
        shutil.copy(__file__, op.join(args.run_path, copied_script_name)) 
    date_ = time.strftime("%Y-%m-%d_%H%M")
    with open(op.join(args.run_path, date_+'_parameters.yml'), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    run_experiments(args)

