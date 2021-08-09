import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os.path as op
from torch.utils.data import TensorDataset, DataLoader


def build_dataloader(x_data, y_data, batch_size, shuffle=True):
        train_data = TensorDataset(torch.from_numpy(x_data).float(), torch.from_numpy(y_data))
        train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size, drop_last=False)
        return train_loader


def epoch_trainer(model, train_loader, optimizer, criterion, logger, device):
    model.train()
    _losses, pred_list, label_list = [], [], []
    m = nn.Softmax(dim=-1)
    for X, label in train_loader:
        model.zero_grad()
        out = model(X.float().to(device))
        label = label.view(-1).long()
        loss = criterion(out, label.to(device))
        _losses.append(loss.item())
        loss.backward()
        optimizer.step()
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label.cpu().detach().numpy().reshape(-1))
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    acc = accuracy_score(y_true, y_pred)
    train_loss = np.average(_losses)
    logger.add_scalar('train_loss', train_loss)
    logger.add_scalar('train_acc', acc)
    return train_loss, acc


def epoch_validation(model, valid_loader, logger, device):
    pred_list, label_list, _losses = [], [], []
    model.eval()
    m = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for x, label in valid_loader:
        out = model(x.float().to(device))
        label = label.view(-1).long()
        loss = criterion(out, label.to(device))
        _losses.append(loss.item())
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label)
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    valid_loss = np.average(_losses)
    acc = accuracy_score(y_true, y_pred)
    logger.add_scalar('valid_loss', valid_loss)
    logger.add_scalar('valid_acc', acc)
    return valid_loss, acc


def create_probs_dataframe(cols, dates, index_list, probs):
    number_cols = len(cols)
    temp_shape_array = np.full((250, number_cols), np.nan)
    r,c = zip(*index_list)
    temp_shape_array[r,c] = probs
    tempdf = pd.DataFrame(temp_shape_array, index=dates, columns=cols)
    best_stocks, worst_stocks = [], []
    for i in range(len(tempdf)):
        best_stocks.append(tempdf.iloc[i].sort_values(ascending=False).dropna()[:10].index.values) 
    df_top = pd.DataFrame(best_stocks, index=tempdf.index)
    return df_top, tempdf



def evaluate_model(model, path, i_sp, device):
    pred_list, label_list = [], []
    probs0, probs1 = [], []
    data_dir = 'data'
    test_x = np.load(op.join(data_dir, 'study_period_X_'+str(i_sp)+'_test.npy'))
    test_y = np.load(op.join(data_dir, 'study_period_Y_'+str(i_sp)+'_test.npy'))
    test_loader = build_dataloader(test_x, test_y, batch_size=5000, shuffle=False)
    
    columns = np.load(op.join(data_dir, 'sp'+str(i_sp),'columns.npy'), allow_pickle=True)
    dates = np.load(op.join(data_dir, 'sp'+str(i_sp),'dates.npy'), allow_pickle=True)
    index_array = np.load(op.join(data_dir, 'sp'+str(i_sp),'index_array.npy'), allow_pickle=True)
    model.eval()
    m = nn.Softmax(dim=-1)
    for x, label in test_loader:
        out = model(x.float().to(device))
        out = m(out)
        probs0.append(out[:,0].cpu().detach().numpy())
        probs1.append(out[:,1].cpu().detach().numpy())
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label)
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    probs0=np.concatenate(probs0)
    probs1=np.concatenate(probs1)

    yhat0_df, probsdf0 = create_probs_dataframe(cols=columns, dates=dates, index_list=index_array, probs=probs0)
    yhat1_df, probsdf1 = create_probs_dataframe(cols=columns, dates=dates, index_list=index_array, probs=probs1)
    yhat0_df.to_csv(path+'/df_y_hat_k_'+str(0)+'.csv')
    yhat1_df.to_csv(path+'/df_y_hat_k_'+str(1)+'.csv')
    probsdf0.to_csv(path+'/df_y_prob_k_'+str(0)+'.csv')
    probsdf1.to_csv(path+'/df_y_prob_k_'+str(1)+'.csv')
    y_hat_df = pd.DataFrame(y_true, columns=['y_true']).join(pd.DataFrame(y_pred, columns=['y_pred']))
    y_hat_df.to_csv(op.join(path, 'df_eval.csv'))

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics = {}
    metrics['acc'] = acc
    metrics['prec'] = prec
    metrics['recall'] = rec
    metrics['f1'] = f1
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    df_metrics.to_csv(op.join(path, 'metrics.csv'))
    return df_metrics

