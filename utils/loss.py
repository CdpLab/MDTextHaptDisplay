import torch
import numpy as np


def get_MASK(labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    return mask


def get_loss_MSE(preds, labels, null_val=np.nan):
    mask = get_MASK(labels=labels, null_val=null_val)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def get_loss_RMSE(preds, labels, null_val=np.nan):
    return torch.sqrt(get_loss_MSE(preds=preds, labels=labels, null_val=null_val))


def get_loss_MAE(preds, labels, null_val=np.nan):
    mask = get_MASK(labels=labels, null_val=null_val)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def get_loss_MAPE(preds, labels, null_val=np.nan):
    mask = get_MASK(labels=labels, null_val=null_val)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels) * 100
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def get_loss(preds, labels):
    mse = get_loss_MSE(preds, labels)
    rmse = get_loss_RMSE(preds, labels)
    mae = get_loss_MAE(preds, labels)
    mape = get_loss_MAPE(preds, labels)
    return mse, rmse, mae, mape
