import torch
import os
import torchvision.transforms as transforms

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor



### flip and brightress
class Augment_RGB_torch_nocrop:
    def __init__(self):
        pass
    def transform0_nocrop(self, torch_tensor):
        return torch_tensor
    def transform1_nocrop(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform2_nocrop(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-1)
        return torch_tensor
    def transform3_nocrop(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-1,-2)
        return torch_tensor
    def transform4_nocrop(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5_nocrop(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-1)
        return torch_tensor
    def transform6_nocrop(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7_nocrop(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-1,-2)
        return torch_tensor

### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1)
        if rgb_noisy.is_cuda is True:
            lam = lam.cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy
