import os
import sys
import argparse
import options
import utils
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime


from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from utils.loader import  get_training_data,get_validation_data
from tensorboardX import SummaryWriter
from torchvision import utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr_val
from skimage.metrics import structural_similarity as ssim_val
from criteria.lpips.lpips import LPIPS
import criteria.piq as piq
import math

def main():
    # add dir
    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(dir_name)
    print(dir_name)
    ######### parser ###########
    opt = options.Options().init(argparse.ArgumentParser()).parse_args()
    print(opt)
    ######### Logs dir ###########
    log_dir = os.path.join(dir_name,'log', opt.env)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logname = os.path.join(log_dir,'log.log')
    print("Now time is : ",datetime.datetime.now().isoformat())
    with open(logname,'a') as f:
        f.write(f"Now time is : {datetime.datetime.now().isoformat()}\n")
    result_dir = os.path.join(log_dir, 'results')
    model_dir  = os.path.join(log_dir, 'models')
    tensorboard_dir  = os.path.join(log_dir, 'tensorboard')
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    ######### Model ###########
    model_restoration = utils.get_arch(opt)
    print(f'model param {(sum(param.numel() for param in  model_restoration.parameters()))/1e6}M')
    with open(logname,'a') as f:
        f.write(f'model param {(sum(param.numel() for param in  model_restoration.parameters()))/1e6}M')
        f.write(str(opt)+'\n')
        f.write(str(model_restoration)+'\n')

    ######### Optimizer ###########
    start_epoch = 1
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(opt.beta1, opt.beta2),eps=1e-8, weight_decay=opt.weight_decay)
    else:
        raise Exception("Error optimizer...")


    ######### DataParallel ###########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
    print('gpu ids:',gpu_ids)
    model_restoration = torch.nn.DataParallel(model_restoration.to('cuda:0'),device_ids=gpu_ids)

    ######### Resume ###########
    if opt.resume:
        path_chk_rest = opt.pretrain_weights
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        lr = utils.load_optim(optimizer, path_chk_rest)

        for p in optimizer.param_groups: p['lr'] = lr
        warmup = False
        new_lr = lr
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:",new_lr)
        print('------------------------------------------------------------------------------')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

    ######### Scheduler ###########
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()




    ######### Initialize Loss ###########
    loss_all = {}
    if opt.lpips_lambda > 0:
        lpips_loss = LPIPS(net_type='alex').to(device).eval()
        loss_all['lpips_loss'] = lpips_loss
    if opt.ssim_lambda > 0:
        ssim_loss = SSIMLoss().to(device)
        loss_all['ssim_loss'] = ssim_loss
    if opt.l2_lambda > 0:
        l2_loss = torch.nn.MSELoss().to(device)
        loss_all['l2_loss'] = l2_loss
    if opt.l1_lambda > 0:
        l1_loss = torch.nn.L1Loss().to(device)
        loss_all['l1_loss'] = l1_loss



    print(f'Loss function: {loss_all.keys()}')
    with open(logname,'a') as f:
        f.write(f'Loss function: {loss_all.keys()}\n')

    ######### DataLoader ###########
    print('===> Loading datasets')
    img_options_train = {'patch_size':opt.train_ps}
    train_dataset = get_training_data(opt.train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.train_workers, pin_memory=True, drop_last=False)

    val_dataset = get_validation_data(opt.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                            num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print(f"Size of training set: {len_trainset}, size of validation set: {len_valset}")

    with open(logname,'a') as f:
        f.write('===> Loading datasets\n')
        f.write(f"Sizeof training set: {len_trainset}, sizeof validation set: {len_valset} \n")

    ######### validation ###########
    with torch.no_grad():
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            filenames = data_val[2]
            psnr_val_rgb.append(utils.batch_PSNR(input_, target, False).item())
        psnr_val_rgb = sum(psnr_val_rgb)/len_valset
        print('Input & GT (PSNR) -->%.2f dB'%(psnr_val_rgb))

    ######### train ###########
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
    best_psnr = 0
    best_psnr_ssim = 0
    best_epoch = 0
    best_iter = 0
    eval_now = len(train_loader)//opt.val_time
    # eval_now = 1
    print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))
    print(f'total iteration {len(train_loader)*opt.nepoch}')


    torch.cuda.empty_cache()

    #set writer
    writer_dict = {
        'writer': SummaryWriter(tensorboard_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    writer = writer_dict['writer']

    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            optimizer.zero_grad()
            global_steps = writer_dict['train_global_steps']

            target = data[0].cuda()
            input_ = data[1].cuda()
            if opt.datarange == '-11':
                input_ = (input_-0.5)/0.5
                target = (target-0.5)/0.5


            # main forward
            restored = model_restoration(input_)
            if opt.datarange == '-11':
                restored = torch.clamp(restored,-1,1)
            else:
                restored = torch.clamp(restored,0,1)
            loss, loss_dict = calc_loss(opt,loss_all,restored,target,latent=None,gan_image=None)
            log_metrics(loss_dict,writer,global_steps)
            epoch_loss +=loss.item()
            loss.backward()
            optimizer.step()

            #### Show ####
            if (i%100==0) or (global_steps < 1000 and global_steps % 25 == 0):
                if opt.img_vision == True:
                    if opt.datarange == '-11':
                        input_vision = input_.cpu().detach()* 0.5 + 0.5
                        target_vision = target.cpu().detach()* 0.5 + 0.5
                        restored_vison = restored.cpu().detach()* 0.5 + 0.5
                        parse_and_log_images(input_vision, target_vision, restored_vison, gan_image=None,global_step = global_steps,
                                             result_dir=result_dir,title='images_train/', display_count=1)
                    else:
                        parse_and_log_images(input_, target, restored, gan_image=None,global_step = global_steps,
                                             result_dir=result_dir,title='images_train/', display_count=1)
                # torchvision
                if opt.torch_vision == True:
                    gt_rgb = torch.squeeze(target[0, :, :, :])
                    restored_rgb = torch.squeeze(restored[0, :, :, :])
                    input_rgb = torch.squeeze(input_[0, :, :, :])
                    gt_rgb = gt_rgb.detach().cpu().numpy()
                    restored_rgb = restored_rgb.detach().cpu().numpy()
                    input_rgb = input_rgb.detach().cpu().numpy()
                    checkrgb = torch.from_numpy(np.stack((gt_rgb, restored_rgb,input_rgb), axis=0))
                    # checkrgb = checkrgb.unsqueeze(1)
                    img_grid = vutils.make_grid(checkrgb, nrow=3,normalize=True)

                    writer.add_image("train_img___gt___output___input", img_grid, global_steps)

                print_metrics(loss_dict, global_steps,logname)

            #### Evaluation ####
            if (i+1)%eval_now==0 and i>0:
                with torch.no_grad():
                    model_restoration.eval()
                    psnr_val_rgb = []
                    ssim_val_rgb = []
                    for ii, data_val in enumerate((val_loader), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        if opt.datarange == '-11':
                            input_ = (input_-0.5)/0.5
                        target = target.cpu().numpy().squeeze().transpose((1,2,0))
                        filenames = data_val[2]

                        # # pad input image to be a multiple of window_size
                        # _, _, h_old, w_old = input_.size()
                        # h_pad = (h_old // opt.win_size + 1) * opt.win_size - h_old
                        # w_pad = (w_old // opt.win_size + 1) * opt.win_size - w_old
                        # input_ = torch.cat([input_, torch.flip(input_, [2])], 2)[:, :, :h_old + h_pad, :]
                        # input_ = torch.cat([input_, torch.flip(input_, [3])], 3)[:, :, :, :w_old + w_pad]

                        # forward
                        restored = model_restoration(input_)
                        # restored= restored[..., :h_old, :w_old]
                        # trans_img = trans_img[..., :h_old, :w_old]

                        if opt.datarange == '-11':
                            restored = restored*0.5+0.5
                            trans_img = trans_img*0.5+0.5

                        # img_vision
                        if opt.img_vision == True:
                            if ii<=2:
                                input_vision = data_val[1].cpu().detach()
                                target_vision = data_val[0].cpu().detach()
                                restored_vison = restored.cpu().detach()
                                parse_and_log_images(input_vision, target_vision, restored_vison, gan_image=None,global_step = global_steps,
                                                     result_dir=result_dir,title='images_val/', display_count=1)

                        restored = torch.clamp(restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
                        psnr_val_rgb.append(psnr_val(restored, target))
                        ssim_val_rgb.append(ssim_val(restored, target, multichannel=True))


                        #torchvision
                        if opt.torch_vision == True:
                            if ii == 0:
                                val_restored_rgb = np.transpose(restored,(2,0,1))
                                val_gt_rgb = np.transpose(target,(2,0,1))
                                val_input = data_val[1].detach().numpy().squeeze()
                                val_checkrgb = torch.from_numpy(np.stack((val_gt_rgb, val_restored_rgb,val_input), axis=0))
                                val_img_grid = vutils.make_grid(val_checkrgb, nrow=3)
                                writer.add_image("val_img___gt___output___input", val_img_grid, global_steps)

                    psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                    ssim_val_rgb = sum(ssim_val_rgb)/len_valset
                    writer.add_scalar('psnr_val', psnr_val_rgb, global_steps)
                    writer.add_scalar('ssim_val', ssim_val_rgb, global_steps)



                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_psnr_ssim = ssim_val_rgb
                        best_epoch = epoch
                        best_iter = i
                        torch.save({'epoch': epoch,
                                    'state_dict': model_restoration.state_dict(),
                                    # 'optimizer' : optimizer.state_dict()
                                    }, os.path.join(model_dir,"model_best.pth"))

                    print("[Epoch %d/%d] [iter %d/%d] [PSNR %.2f] [SSIM: %.4f] ----  [Best_Ep %d] [Best_it %d] [Best_PSNR %.2f], [P_SSIM %.4f] " % (epoch,opt.nepoch + 1, i,len(train_loader), psnr_val_rgb,ssim_val_rgb, best_epoch,best_iter,best_psnr,best_psnr_ssim))
                    with open(logname,'a') as f:
                        f.write("[Epoch %d/%d] [iter %d/%d] [PSNR %.2f] [SSIM: %.4f] ----  [Best_Ep %d] [Best_it %d] [Best_PSNR %.2f], [P_SSIM %.4f] " \
                                % (epoch, opt.nepoch + 1,i,len(train_loader), psnr_val_rgb,ssim_val_rgb,best_epoch,best_iter,best_psnr,best_psnr_ssim)+'\n')
                    model_restoration.train()
                    torch.cuda.empty_cache()
            writer_dict['train_global_steps'] = global_steps + 1
        scheduler.step()


        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        with open(logname,'a') as f:
            f.write("------------------------------------------------------------------"+'\n')
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')
            f.write("------------------------------------------------------------------"+'\n')

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    # 'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))
        #lr
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

    print("Now time is : ",datetime.datetime.now().isoformat())
    with open(logname,'a') as f:
        f.write(f"Now time is : {datetime.datetime.now().isoformat()}")

def parse_and_log_images(x, y, y_hat, gan_image,global_step,result_dir,title=None, subscript=None, display_count=1):
    im_data = []
    for i in range(display_count):
        if gan_image is not None:
            cur_im_data = {
                'input': tensor2im(x[i]),
                'target': tensor2im(y[i]),
                'output': tensor2im(y_hat[i]),
                'gan_image' :tensor2im(gan_image[i])
            }
        else:
            cur_im_data = {
                'input': tensor2im(x[i]),
                'target': tensor2im(y[i]),
                'output': tensor2im(y_hat[i])
            }

        im_data.append(cur_im_data)
    log_images(title, global_step,result_dir,im_data=im_data, subscript=subscript)



def log_images( name,global_step, result_dir,im_data, subscript=None):
    fig = vis_img(im_data)
    step = global_step
    if subscript:
        path = os.path.join(result_dir, name, f'{subscript}_{step:04d}.jpg')
    else:
        path = os.path.join(result_dir, name, f'{step:04d}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close('all')

def vis_img(log_hooks):
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(16, 8 * display_count))
    row = len(log_hooks[0])
    gs = fig.add_gridspec(display_count, row)
    for i in range(display_count):
        hooks_dict = log_hooks[i]
        vis_img_show(hooks_dict, fig, gs, i)
    fig.tight_layout()
    return fig

def vis_img_show(hooks_dict, fig, gs, i):
    ax1 = fig.add_subplot(gs[i, 0])
    ax1.imshow(hooks_dict['input'])
    ax1.set_title('Input')

    ax2 = fig.add_subplot(gs[i, 1])
    ax2.imshow(hooks_dict['target'])
    ax2.set_title('Target')

    ax3 = fig.add_subplot(gs[i, 2])
    ax3.imshow(hooks_dict['output'])
    ax3.set_title('Output')

    if 'gan_image' in hooks_dict.keys():
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.imshow(hooks_dict['gan_image'])
        ax4.set_title('trans_image')


def tensor2im(var):

    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def calc_loss(opt,loss_all, y_hat, y, latent=None,gan_image=None):
    loss_dict = {}
    loss = 0.0

    if opt.l2_lambda > 0:
        loss_l2 = loss_all['l2_loss'](y_hat, y)
        loss_dict['loss_l2'] = float(loss_l2)
        loss += loss_l2 * opt.l2_lambda

    if opt.lpips_lambda > 0:
        loss_lpips = loss_all['lpips_loss'](y_hat, y)
        loss_dict['loss_lpips'] = float(loss_lpips)
        loss += loss_lpips * opt.lpips_lambda

    if opt.ssim_lambda > 0:
        # loss_ssim = loss_all['ssim_loss']((y_hat+1.)/2., ((y+1.)/2.))
        loss_ssim = loss_all['ssim_loss'](y_hat, y)
        loss_dict['loss_ssim'] = float(loss_ssim)
        loss += loss_ssim * opt.ssim_lambda

    if opt.l1_lambda > 0:
        loss_l1 = loss_all['l1_loss'](y_hat, y)
        loss_dict['loss_l1'] = float(loss_l1)
        loss += loss_l1 * opt.l1_lambda


    loss_dict['loss'] = float(loss)

    return loss, loss_dict

def print_metrics(metrics_dict, global_step,logname):
    with open(logname,'a') as f:
        print(f'Metrics step {global_step}')
        f.write(f'Metrics step {global_step}\n')
        for key, value in metrics_dict.items():
            print(f'\t{key} = {value}')
            f.write(f'\t{key} = {value}\n')


def log_metrics(metrics_dict, writer,global_step):
    for key, value in metrics_dict.items():
        writer.add_scalar(f'{key}', value, global_step)

# SSIM Loss
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss,self).__init__()
        self.loss = piq.SSIMLoss()

    def forward(self, sr, hr):
        return self.loss(sr, hr)

if __name__ == '__main__':
    main()
