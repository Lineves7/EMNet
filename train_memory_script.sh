C:/Anaconda3/envs/pytorch181/python train_memory.py \
--env emnet_enhancer_lolv1 \
--arch emnet_enhancer \
--arch_memory emnet_memory \
--pretrain_weights ./pre_trained_logs/enhancer/model_lolv1.pth \
--embed_dim 32 \
--nepoch 8000 \
--train_dir C:\\dataset\\data\\LOL\\LOLdataset\\our485 \
--val_dir C:\\dataset\\data\\LOL\\LOLdataset\\eval15 \
--train_workers 4 \
--datarange 01 \
--K_feat_dim 512 \
--V_feat_dim 1 \
--thres 0.00001 \
--mem_size 9000 \
--pooling_size 1 \
--top_k 32 \
--testing_top_k 1 \
--val_time 1 \
--train_ps 128 \
--batch_size 128 \
--cos_thres 0.008
