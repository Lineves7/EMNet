import torch
import numpy as np
import pickle
import cv2
import math

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif','bmp'])

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.

    return img

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1.0)

    return img, mask

def img2patches(imgs,patch_size:tuple,stride_size:tuple):
    """
    Args:
        imgs: (H,W)/(H,W,C)/(B,H,W,C)
        patch_size: (patch_h, patch_w)
        stride_size: (stride_h, stride_w)
    """

    if imgs.ndim == 2:
        # (H,W) -> (1,H,W,1)
        imgs = np.expand_dims(imgs,axis=2)
        imgs = np.expand_dims(imgs,axis=0)
    elif imgs.ndim == 3:
        # (H,W,C) -> (1,H,W,C)
        imgs = np.expand_dims(imgs,axis=0)
    b,h,w,c = imgs.shape
    p_h,p_w = patch_size
    s_h,s_w = stride_size

    assert (h-p_h) % s_h == 0 and (w-p_w) % s_w == 0

    n_patches_y = (h - p_h) // s_h + 1
    n_patches_x = (w - p_w) // s_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    n_patches = n_patches_per_img * b
    patches = np.empty((n_patches,p_h,p_w,c),dtype=imgs.dtype)

    patch_idx = 0
    for img in imgs:
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * s_h
                y2 = y1 + p_h
                x1 = j * s_w
                x2 = x1 + p_w
                patches[patch_idx] = img[y1:y2, x1:x2]
                patch_idx += 1
    return patches

def unpatch2d(patches, imsize: tuple, stride_size: tuple):
    '''
        patches: (n_patches, p_h, p_w,c)
        imsize: (img_h, img_w)
    '''
    assert len(patches.shape) == 4

    i_h, i_w = imsize
    n_patches,p_h,p_w,c = patches.shape
    s_h, s_w = stride_size

    assert (i_h - p_h) % s_h == 0 and (i_w - p_w) % s_w == 0

    n_patches_y = (i_h - p_h) // s_h + 1
    n_patches_x = (i_w - p_w) // s_w + 1
    n_patches_per_img = n_patches_y * n_patches_x
    batch_size = n_patches // n_patches_per_img

    imgs = np.zeros((batch_size,i_h,i_w,c))
    weights = np.zeros_like(imgs)


    for img_idx, (img,weights) in enumerate(zip(imgs,weights)):
        start = img_idx * n_patches_per_img

        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y1 = i * s_h
                y2 = y1 + p_h
                x1 = j * s_w
                x2 = x1 + p_w
                patch_idx = start + i*n_patches_x+j
                img[y1:y2,x1:x2] += patches[patch_idx]
                weights[y1:y2, x1:x2] += 1
    imgs /= weights

    return imgs.astype(patches.dtype)