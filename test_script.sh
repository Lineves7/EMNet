C:/Anaconda3/envs/pytorch181/python test.py \
--arch emnet_enhancer \
--arch_memory emnet_memory \
--pooling_size 1 \
--mem_size 9000 \
--embed_dim 32 \
--V_feat_dim 1 \
--result_dir ./log_eval/lolv1 \
--embed_dim 32 \
--weights ./pre_trained_logs/enhancer/model_lolv1.pth \
--mem_weights ./pre_trained_logs/memory/memory_lolv1.pth \
--input_dir C:\\dataset\\data\\LOL\\LOLdataset\\eval15 \
--testing_top_k 1 \
--save_images

C:/Anaconda3/envs/pytorch181/python test.py \
--arch emnet_enhancer \
--arch_memory emnet_memory \
--pooling_size 1 \
--mem_size 9000 \
--embed_dim 32 \
--V_feat_dim 1 \
--result_dir ./log_eval/lolv2 \
--embed_dim 32 \
--weights ./pre_trained_logs/enhancer/model_lolv2.pth \
--mem_weights ./pre_trained_logs/memory/memory_lolv2.pth \
--input_dir C:\\dataset\\data\\LOL-v2\\Real_captured\\Test \
--testing_top_k 1 \
--save_images

