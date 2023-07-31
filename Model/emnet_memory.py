import torch
import torch.nn as nn
import numbers

import functools
from einops import rearrange
from torch.nn import Module
from collections.abc import Iterable
from torch.nn import functional as F
from torchvision import models




class MemoryNet(nn.Module):
    def __init__(self,mem_size = 1024,V_feat_dim = 1, K_feat_dim = 512, top_k = 256, alpha = 0.001, age_noise = 4.0):
        super(MemoryNet, self).__init__()
        self.ResNet18 = ResNet18()
        self.ResNet18 = self.ResNet18.eval()
        self.mem_size = mem_size
        self.V_feat_dim = V_feat_dim
        self.K_feat_dim = K_feat_dim
        self.alpha = alpha
        self.age_noise = age_noise
        self.top_k = top_k

        self.value = random_uniform((self.mem_size, self.V_feat_dim), 0, 1).cuda()

        self.key = F.normalize(random_uniform((self.mem_size, self.K_feat_dim), -0.01, 0.01), dim=1).cuda()
        self.age = torch.zeros(self.mem_size).cuda()

        self.top_index = torch.zeros(self.mem_size).cuda()
        self.top_index = self.top_index - 1.0
        self.key.requires_grad = False


        self.body = [self.ResNet18]
        self.body = nn.Sequential(*self.body)


    def forward(self, x):
        q = self.body(x)
        q = F.normalize(q, dim = 1)
        return q


    def memory_update(self, query, color_feat, color_thres, top_index):

        cosine_score = torch.matmul(query, torch.t(self.key))
        top1_score, top1_index = torch.topk(cosine_score, 1, dim = 1)
        top1_index = top1_index[:, 0]
        top1_color_value = self.value[top1_index]

        color_similarity = torch.squeeze(torch.abs(top1_color_value - color_feat))
        if len(color_similarity.shape)>1:
            color_similarity = torch.mean(color_similarity,1)

        memory_mask = color_similarity < color_thres
        self.age = self.age + 1.0

        ## Case 1
        case_index = top1_index[memory_mask]
        self.key[case_index] = F.normalize(self.key[case_index] + query[memory_mask], dim = 1)
        self.age[case_index] = 0.0

        ## Case 2
        memory_mask = ~memory_mask
        case_index = top1_index[memory_mask]

        random_noise = random_uniform((self.mem_size, 1), -self.age_noise, self.age_noise)[:, 0]
        random_noise = random_noise.cuda()
        age_with_noise = self.age + random_noise
        old_values, old_index = torch.topk(age_with_noise, len(case_index), dim=0)

        self.key[old_index] = query[memory_mask]
        self.value[old_index] = color_feat[memory_mask]
        self.top_index[old_index] = top_index[memory_mask]
        self.age[old_index] = 0.0


    def topk_feature(self, query, top_k = 1):
        _bs = query.size()[0]
        cosine_score = torch.matmul(query, torch.t(self.key))

        topk_score, topk_index = torch.topk(cosine_score, top_k, dim = 1)
        weight = torch.nn.functional.softmax(topk_score)

        topk_feat = torch.cat([torch.unsqueeze(self.value[topk_index[i], :], dim = 0) for i in range(_bs)], dim = 0)
        topk_idx = torch.cat([torch.unsqueeze(self.top_index[topk_index[i]], dim = 0) for i in range(_bs)], dim = 0)

        # topk_feat = torch.matmul(weight,topk_feat)
        return topk_feat, topk_idx


    def _unsupervised_loss(self, pos_score, neg_score):

        hinge = torch.clamp(neg_score - pos_score + self.alpha, min = 0.0)
        loss = torch.mean(hinge)

        return loss

def random_uniform(shape, low, high):
    x = torch.rand(*shape)
    result = (high - low) * x + low

    return result



def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = freeze

class ResNet18(nn.Module):
    def __init__(self, pre_trained = True, require_grad = False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained = True)

        self.body = [layers for layers in self.model.children()]
        self.body.pop(-1)

        self.body = nn.Sequential(*self.body)

        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 512)
        return x
