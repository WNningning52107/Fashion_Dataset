import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd
from torch.utils import data
import torch.distributed as dist

from model.op import conv2d_gradfix
from utils.linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, my_merge_fill, show_fill_map
from utils.linefiller.thinning import thinning


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def squeeze_label_map(label_map):
    ret_label_map = label_map.copy()
    
    labels, counts = np.unique(ret_label_map, return_counts=True)
    label_orders = np.argsort(counts)
    
    for ord_id, ord_val in enumerate(label_orders):
        mask = (label_map == labels[ord_val])
        ret_label_map[mask] = ord_id
    
    return ret_label_map


def get_map(binary, in_image=None, do_merge=True):
    fills = []
    result = binary

    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)
    # print('result num 3: ', len(fills))
    
    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)
    # print('result num 2: ', len(fills))
    
    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)
    # print('result num 1: ', len(fills))

    fill = flood_fill_multi(result)
    fills += fill
    # print('flood_fill_multi num 1: ', len(fills))

    fillmap = build_fill_map(result, fills)

    if do_merge:
        if in_image is None:
            fillmap = merge_fill(fillmap, max_iter=10)
        else:
            fillmap = my_merge_fill(in_image, fillmap)
    
    fillmap = thinning(fillmap)
    fillmap = squeeze_label_map(fillmap)
    return fillmap

def get_map2(pic):
    fills = []
    result = pic
    fill = trapped_ball_fill_multi(result, radius=1, method='percentage', percent=0.0001)
    fills += fill
    result = mark_fill(result, fill)
    fillmap = build_fill_map(result, fills)
    fillmap = merge_fill(fillmap)
    fillmap = thinning(fillmap)
    fillmap = squeeze_label_map(fillmap)
    return fillmap
