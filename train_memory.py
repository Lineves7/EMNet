import os
import sys
import argparse
import options
import utils
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn

from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from tqdm import tqdm


from utils.loader import  get_training_data,get_validation_data
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr_val
from skimage.metrics import structural_similarity as ssim_val
import torch.nn.functional as F
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

    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration,path_chk_rest)
    print('------------------------------------------------------------------------------')
    print("==> Resuming model_restoration weight from:",opt.pretrain_weights)
    print('------------------------------------------------------------------------------')

    model_restoration_mem = utils.get_arch_mem(opt)


    ######### DataParallel ###########
    gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
    print('gpu ids:',gpu_ids)
    model_restoration = torch.nn.DataParallel(model_restoration.to('cuda:0'),device_ids=gpu_ids)
    model_restoration_mem = torch.nn.DataParallel(model_restoration_mem.to('cuda:0'),device_ids=gpu_ids)


    ######### DataLoader ###########
    print('===> Loading datasets')
    img_options_train = {'patch_size':opt.train_ps}
    train_dataset = get_training_data(opt.train_dir,img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.train_workers, pin_memory=True, drop_last=True)

    val_dataset = get_validation_data(opt.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                            num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print(f"Sizeof training set: {len_trainset}, sizeof validation set: {len_valset}")

    with open(logname,'a') as f:
        f.write('===> Loading datasets\n')
        f.write(f"Size of training set: {len_trainset}, size of validation set: {len_valset} \n")


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
    start_epoch = 1
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
    best_psnr = 0
    best_psnr_ssim = 0
    best_psnr_mem = 0
    best_psnr_ssim_mem = 0
    best_epoch = 0
    best_iter = 0
    best_epoch_mem = 0
    best_iter_mem = 0
    ori_psnr = 0

    eval_now = len(train_loader)//opt.val_time
    # eval_now = 1
    print(f"\nEvaluation after every {eval_now} Iterations !!!\n")
    print(f'total iteration {len(train_loader)*opt.nepoch}')
    with open(logname,'a') as f:
        f.write(f"\nEvaluation after every {eval_now} Iterations !!!\n")
        f.write(f'total iteration {len(train_loader)*opt.nepoch}')


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
            global_steps = writer_dict['train_global_steps']
            target = data[0].cuda()
            input_ = data[1].cuda()
            idx = data[4].cuda()
            input_mem = data[5].cuda()

            # memory update
            with torch.no_grad():
                mem_q = model_restoration_mem(input_mem)
                avg_pooling = nn.AdaptiveAvgPool2d((opt.pooling_size,opt.pooling_size))
                v_feat_gt = avg_pooling(target)
                if opt.pooling_mean == True:
                    v_feat_gt = torch.mean(v_feat_gt,1).unsqueeze(1)
                v_feat_gt = v_feat_gt.view(v_feat_gt.shape[0],-1).cuda()
                model_restoration_mem.module.memory_update(mem_q,v_feat_gt,opt.thres,idx)

            #### Evaluation ####
            if (i+1)%eval_now==0 and i>=0:
                with torch.no_grad():
                    model_restoration.eval()
                    if ori_psnr == 0:
                        psnr_val_rgb = []
                        ssim_val_rgb = []
                    psnr_val_rgb_ft = []
                    ssim_val_rgb_ft = []
                    psnr_val_rgb_mem = []
                    ssim_val_rgb_mem = []
                    psnr_val_rgb_ft_mean = []

                    for ii, data_val in enumerate((val_loader), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        input_mem = data_val[5].cuda()
                        if opt.datarange == '-11':
                            input_ = (input_-0.5)/0.5
                        target = target.cpu().numpy().squeeze().transpose((1,2,0))
                        filenames = data_val[2]

                        # pad input image to be a multiple of window_size
                        # _, _, h_old, w_old = input_.size()
                        # h_pad = (h_old // opt.win_size + 1) * opt.win_size - h_old
                        # w_pad = (w_old // opt.win_size + 1) * opt.win_size - w_old
                        # input_ = torch.cat([input_, torch.flip(input_, [2])], 2)[:, :, :h_old + h_pad, :]
                        # input_ = torch.cat([input_, torch.flip(input_, [3])], 3)[:, :, :, :w_old + w_pad]

                        # forward
                        restored = model_restoration(input_)
                        # restored= restored[..., :h_old, :w_old]


                        if opt.datarange == '-11':
                            restored = restored*0.5+0.5


                        # test_gt_avgpooling
                        mean_out = avg_pooling(restored)
                        mean_gt = avg_pooling(data_val[0].cuda())
                        if opt.pooling_mean == True:
                            mean_out = torch.mean(mean_out,1).unsqueeze(1)
                            mean_gt = torch.mean(mean_gt,1).unsqueeze(1)

                        ratio_map = (mean_gt/mean_out)
                        ratio_map_up = torch.nn.functional.interpolate(ratio_map, size=[restored.shape[2],restored.shape[3]],mode='nearest')
                        restored_ft = torch.clamp(torch.clamp(restored, 0, 1)*ratio_map_up,0,1)
                        restored_ft = restored_ft.cpu().numpy().squeeze().transpose((1,2,0))
                        # change to uint8
                        restored_ft = np.clip((restored_ft * 255.0).round(), 0, 255)
                        target = (target*255.0)
                        psnr_val_rgb_ft.append(psnr_val(restored_ft, target,data_range=255))
                        ssim_val_rgb_ft.append(ssim_val(restored_ft, target,  data_range=255,multichannel=True))

                        # memory
                        query = model_restoration_mem(input_mem)
                        top1_feature, _ = model_restoration_mem.module.topk_feature(query, opt.testing_top_k)
                        top1_feature = top1_feature[:, 0, :].squeeze()

                        # test
                        mean_out_mem = top1_feature
                        if opt.pooling_mean == True:
                            mean_out_mem = mean_out_mem.view(1,1,opt.pooling_size,opt.pooling_size)
                        else:
                            mean_out_mem = mean_out_mem.view(1,3,opt.pooling_size,opt.pooling_size)
                        ratio_map = (mean_out_mem/mean_out)
                        mean_cat = torch.cat((mean_out,mean_out_mem))
                        weight = torch.nn.functional.softmax(mean_cat,dim=0)
                        alpha1 = weight[0]
                        alpha2 = weight[1]
                        ratio_map_up = alpha1 + alpha2*ratio_map


                        restored_mem = torch.clamp(torch.clamp(restored, 0, 1)*ratio_map_up,0,1)

                        restored_mem = restored_mem.cpu().numpy().squeeze().transpose((1,2,0))


                        # ori change to uint8
                        if ori_psnr == 0:
                            restored = torch.clamp(restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
                            restored = np.clip((restored * 255.0).round(), 0, 255)
                            psnr_val_rgb.append(psnr_val(restored, target,data_range=255))
                            ssim_val_rgb.append(ssim_val(restored, target,data_range=255,multichannel=True))

                        # mem change to uint8
                        restored_mem = np.clip((restored_mem * 255.0).round(), 0, 255)
                        psnr_val_rgb_mem.append(psnr_val(restored_mem, target,data_range=255))
                        ssim_val_rgb_mem.append(ssim_val(restored_mem, target, data_range=255,multichannel=True))



                    psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                    ssim_val_rgb = sum(ssim_val_rgb)/len_valset
                    psnr_val_rgb_ft = sum(psnr_val_rgb_ft)/len_valset
                    ssim_val_rgb_ft = sum(ssim_val_rgb_ft)/len_valset
                    psnr_val_rgb_mem = sum(psnr_val_rgb_mem)/len_valset
                    ssim_val_rgb_mem = sum(ssim_val_rgb_mem)/len_valset

                    writer.add_scalar('psnr_val', psnr_val_rgb, global_steps)
                    writer.add_scalar('ssim_val', ssim_val_rgb, global_steps)
                    writer.add_scalar('psnr_val_ft', psnr_val_rgb_ft, global_steps)
                    writer.add_scalar('ssim_val_ft', ssim_val_rgb_ft, global_steps)
                    writer.add_scalar('psnr_val_mem', psnr_val_rgb_mem, global_steps)
                    writer.add_scalar('ssim_val_mem', ssim_val_rgb_mem, global_steps)

                    if psnr_val_rgb_mem > best_psnr_mem:
                        best_psnr_mem = psnr_val_rgb_mem
                        best_psnr_ssim_mem = ssim_val_rgb_mem
                        best_epoch_mem = epoch

                        best_iter_mem = i
                        if model_restoration_mem.module.value is not None:
                            torch.save({'epoch': epoch,
                                        'mem_key' : model_restoration_mem.module.key.cpu(),
                                        'mem_vale' : model_restoration_mem.module.value.cpu(),
                                        'mem_age' : model_restoration_mem.module.age.cpu(),
                                        'mem_index' : model_restoration_mem.module.top_index.cpu(),
                                        }, os.path.join(model_dir,"model_best_mem.pth"))
                        else:
                            print('model_restoration_mem.module.value is None')

                    print(f"[Epoch {epoch}/{opt.nepoch + 1}] [iter {i}/{len(train_loader)}] [PSNR {psnr_val_rgb:.2f}] [SSIM: {ssim_val_rgb:.4f}]   [PSNR_mem {psnr_val_rgb_mem:.2f}] [SSIM_mem: {ssim_val_rgb_mem:.4f}]  [PSNR_gt {psnr_val_rgb_ft:.2f}] [SSIM_gt: {ssim_val_rgb_ft:.4f}] ---- [Best_Ep {best_epoch}] [Best_PSNR {best_psnr:.2f}], [P_SSIM {best_psnr_ssim:.4f}] ---- [Best_Ep_mem {best_epoch_mem}] [Best_PSNR_mem {best_psnr_mem:.2f}], [P_SSIM_mem {best_psnr_ssim_mem:.4f}]")
                    with open(logname,'a') as f:
                        f.write(f"[Epoch {epoch}/{opt.nepoch + 1}] [iter {i}/{len(train_loader)}] [PSNR {psnr_val_rgb:.2f}] [SSIM: {ssim_val_rgb:.4f}] [PSNR_mem {psnr_val_rgb_mem:.2f}] [SSIM_mem: {ssim_val_rgb_mem:.4f}] [PSNR_gt {psnr_val_rgb_ft:.2f}] [SSIM_gt: {ssim_val_rgb_ft:.4f}] ---- [Best_Ep {best_epoch}] [Best_PSNR {best_psnr:.2f}], [P_SSIM {best_psnr_ssim:.4f}] ---- [Best_Ep_mem {best_epoch_mem}] [Best_PSNR_mem {best_psnr_mem:.2f}], [P_SSIM_mem {best_psnr_ssim_mem:.4f}]\n")
                    model_restoration.train()
                    torch.cuda.empty_cache()
            writer_dict['train_global_steps'] = global_steps + 1


        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time,epoch_loss))
        print("------------------------------------------------------------------")
        with open(logname,'a') as f:
            f.write("------------------------------------------------------------------"+'\n')
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time,epoch_loss)+'\n')
            f.write("------------------------------------------------------------------"+'\n')

        torch.save({'epoch': epoch,
                    'mem_key' : model_restoration_mem.module.key.cpu(),
                    'mem_vale' : model_restoration_mem.module.value.cpu(),
                    'mem_age' : model_restoration_mem.module.age.cpu(),
                    'mem_index' : model_restoration_mem.module.top_index.cpu(),
                    }, os.path.join(model_dir,"model_latest.pth"))


    print("Now time is : ",datetime.datetime.now().isoformat())
    with open(logname,'a') as f:
        f.write(f"Now time is : {datetime.datetime.now().isoformat()}")



if __name__ == '__main__':
    main()
