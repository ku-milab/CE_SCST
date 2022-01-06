from utils import *
from config import *
import model
import numpy as np
from sklearn.metrics import f1_score
import torch
from config import *
from scipy.stats import entropy
import torch.nn.functional as F


def trainer(tr_loader, network, opt, loss_weight=None, Y_t=None, log_file=None):
    Y_tr_hat, Y_tr_new = np.array([]), np.array([])

    loss_per_ep = 0
    for (x, y), idx in tr_loader:
        x, y, y_t = x.to(device), y.flatten().to(device), Y_t[idx].flatten().to(device)

        opt.opt_zero_grad()


        x = network.FE(x)
        x_att, l_1 = network.stage_confusion_estimator(x)
        l_2, l_2_t = network.CE(x, x_att)


        l_1 = l_1.flatten(end_dim=1)
        l_2 = l_2.flatten(end_dim=1)
        l_2_t = l_2_t.flatten(end_dim=1)
        loss_1 = loss_cross_entropy(weight=loss_weight['org'])(l_1, y)
        loss_2 = loss_cross_entropy(weight=loss_weight['org'])(l_2, y)
        loss_t = loss_cross_entropy(weight=loss_weight['trans'])(l_2_t, y_t)
        loss = loss_1.mean() + 2*loss_2.mean() + loss_t.mean()    # cls 1st, task aux
        loss.backward()
        opt.opt_step()

        # For Record
        y_hat = dcn(l_2.detach().argmax(-1))
        Y_tr_hat = np.concatenate([Y_tr_hat, y_hat])
        Y_tr_new = np.concatenate([Y_tr_new, dcn(y)])
        loss_per_ep += dcn(loss).mean()

    loss_per_ep /= len(tr_loader)
    if args.scheme=='M_M': Y_tr_new = Y_tr_new.flatten()
    f1_tr = round(f1_score(dcn(Y_tr_new), Y_tr_hat, average='macro', zero_division=1), 4)
    return loss_per_ep, f1_tr















