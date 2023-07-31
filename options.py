import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings

        



        # args for model
        parser.add_argument('--arch', type=str, default ='emnet_enhancer',  help='archtechture of image enhancer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')


        
        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--train_dir', type=str, default ='../LOL-v2/Real_captured/Train/',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='../LOL-v2/Real_captured/Test/',  help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup')
        parser.add_argument('--exp_name',type=str,help='The name of exp')
        parser.add_argument('--val_time', type=int, default=2, help='val time in per epoch')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=2, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=2, help='eval_dataloader workers')
        parser.add_argument('--pretrain_weights',type=str, default='./pre_trained_logs/enhancer/model_lolv2.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--gpu', type=str, default='0,1', help='GPUs')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--beta1',type=float,default=0.9,help='adam: decay of first order momentum of gradient')
        parser.add_argument('--beta2',type=float,default=0.999,help='adam: decay of seconder order momentum of gradient')
        parser.add_argument('--datarange', type=str, default ='-11',  help='The range of input data 01 or -11')
        parser.add_argument('--img_vision', action='store_true', default=False, help='Show intermediate results of jpg in result dir')
        parser.add_argument('--torch_vision', action='store_true', default=False, help='Show intermediate results in tensorbard dir')

        # args for loss weight
        parser.add_argument('--lpips_lambda', default=0, type=float, help='lpips loss weight')
        parser.add_argument('--l2_lambda', default=1, type=float, help='l2 loss weight')
        parser.add_argument('--l1_lambda', default=0, type=float, help='l1 loss weight')
        parser.add_argument('--ssim_lambda', default=0, type=float, help='ssim loss weight')

        # args for saving
        parser.add_argument('--save_dir', type=str, default ='/home/ma-user/work/deNoTr/log',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--save_log', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')



        #args for memory
        parser.add_argument("--mem_size", type = int, default = 1024)
        parser.add_argument("--alpha", type = float, default = 0.1)
        parser.add_argument("--top_k", type = int, default = 256)
        parser.add_argument("--V_feat_dim", type = int, default = 1)
        parser.add_argument("--K_feat_dim", type = int, default = 512)
        parser.add_argument("--thres", type = float, default = 0.0001)
        parser.add_argument('--arch_memory', type=str, default ='emnet_memory',  help='archtechture for memory network')
        parser.add_argument("--pooling_size", type = int, default = 2)
        parser.add_argument("--testing_top_k", type = int, default = 1)
        parser.add_argument('--pooling_mean', action='store_true',default=False,help='avg polling mean')
        parser.add_argument("--resize_size", nargs='+', type = int, default = None)
        parser.add_argument("--cos_thres", type = float, default = 0.001)

        # args for test
        parser.add_argument('--input_dir', default='C:\\dataset\\data\\LOL\\LOLdataset\\eval15',type=str, help='Directory of validation images')
        parser.add_argument('--result_dir', default='./log_eval/test_lolv2',type=str, help='Directory for results')
        parser.add_argument('--weights', default='./pre_trained_logs/enhancer/model_lolv1.pth',type=str, help='Path to weights')
        parser.add_argument('--mem_weights', default='./pre_trained_logs/memory/memory_lolv1.pth',type=str, help='Path to memmoryNet weights')

        return parser
