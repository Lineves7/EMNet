import torch
import torch.nn as nn
import os
from collections import OrderedDict
import Model


def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    arch = opt.arch

    exec('import '+'Model.'+arch.lower())
    model_restoration = eval('Model.'+arch.lower()+'.'+'Enhancer')(opt)

    return model_restoration

def get_arch_mem(opt):

    print('You choose memmory network of '+opt.arch_memory+'. '+'MemoryNet')
    exec('import '+'Model.'+opt.arch_memory.lower())
    opt.V_feat_dim = opt.pooling_size**2*3
    if opt.pooling_mean == True:
        opt.V_feat_dim = opt.pooling_size**2
    print(f'pooling_size is {opt.pooling_size}, V_feat_dim is {opt.V_feat_dim }')
    model_restoration_mem = eval('Model.'+opt.arch_memory.lower()+'.'+'MemoryNet')(mem_size = opt.mem_size,V_feat_dim = opt.V_feat_dim, K_feat_dim = opt.K_feat_dim, top_k = opt.top_k, alpha = opt.alpha, age_noise = 4.0)

    return model_restoration_mem
