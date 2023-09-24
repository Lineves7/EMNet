# Glow in the Dark: Low-Light Image Enhancement with External Memory (TMM 2023)


Official Pytorch implementation of "**Glow in the Dark: Low-Light Image Enhancement with External Memory**" published in ***IEEE Transactions on  Multimedia (TMM)***.
#### [[Paper-official](https://ieeexplore.ieee.org/document/10177254)] 
Dongjie Ye, [Zhangkai Ni](https://eezkni.github.io/),[Wenhan Yang](https://flyywh.github.io/), [Hanli Wang](https://mic.tongji.edu.cn/51/91/c9778a86417/page.htm), [Shiqi Wang](https://www.cs.cityu.edu.hk/~shiqwang/), [Sam Kwong](http://www6.cityu.edu.hk/stfprofile/cssamk.htm)

 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glow-in-the-dark-low-light-image-enhancement/low-light-image-enhancement-on-lol-v2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol-v2?p=glow-in-the-dark-low-light-image-enhancement)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glow-in-the-dark-low-light-image-enhancement/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=glow-in-the-dark-low-light-image-enhancement)

## Requirements and Installation
pytorch=1.8.1

(Please refer EMNet_env.yml for our operational environment.  

**Be aware that EMNet_env.yml includes redundant libraries. Kindly install the relevant libraries according to your needs.**)

## Testing (Running pretrained models)
- Checkpoint

Checkpoints trained on LOL-v1 and LOL-v2 dataset can be found from [Google Drive](https://drive.google.com/file/d/18Fmnlf6qXmH8Op12DsiiH_w-qCZsqKhG/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1RqPGNd65sz009azA2E6F5A?pwd=zhwy ) (提取码：zhwy). 


- Inference
1. Unzip the checkpoint file and place all the files in the ./pre_trained_logs/enhancer/ and ./pre_trained_logs/memory/ directory.
2. Edit the test_script.sh file to modify your python path and the [--input_dir] by specifying the path to your test datasets.
3. Excute the test script below:
    ```
    sh test_script.sh
    ```
4. The results are displayed in the './log_eval/lolv1' and './log_eval/lolv2' directories. In the '[ori]' folder, you can find the original outputs from the image enhancer, while in the '[mem]' folder, you can find the outputs after utilizing external memory.
5. You may also replace our image enhancer with other existing methods if desired.

## Training (Training from scratch)
- ### Image Enhancer

Beforer training the external memory, you need to train the image enhancer first.
1. Edit the train_enhancer_script.sh file to modify your python path and the [--train_dir], [--val_dir] by specifying the path to your training datasets.
2. Excute the training script for image enhancer below:
    ```
    sh train_enhancer_script.sh
    ```
3. Find the trained image enhancer weight in the ./log/[env]/models/ folder.



- ### External Memory
  
Training the external memory requires a pre-trained image enhancer.
1. Edit the train_memory_script.sh file to modify your python path, the [--pretrain_weights] by specifying the path to your pretrained_weights of image enhancer, and the [--train_dir], [--val_dir] by specifying the path to your training datasets.
2.  Excute the training script for external memory below:
    ```
    sh train_memory_script.sh
    ```
3. Find the trained memory weight in the ./log/[env]/models/ folder.

- ### Test
1. Edit the test_script.sh file to modify your python path, [--weights] by specifying the path to your image enhancer, [--mem_weights] by specifying the path to your external memory, and the [--input_dir] by specifying the path to your test datasets.
2. Excute the test script below:
    ```
    sh test_script.sh
    ```
3. The [input_dir] directory structure will be arranged as:
```
[your input dir]
    |- high
        |- 695.png (or alternative img formats)
        |- ...
    |- low
        |- 695.png (or alternative img files)
        |- ...
```

## Citation
If this code is useful for your research, please cite our paper:

```
@article{emnet,
  author={Ye, Dongjie and Ni, Zhangkai and Yang, Wenhan and Wang, Hanli and Wang, Shiqi and Kwong, Sam},
  journal={IEEE Transactions on Multimedia}, 
  title={Glow in the Dark: Low-Light Image Enhancement with External Memory}, 
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TMM.2023.3293736}}
```

## Contact

Thanks for your attention! If you have any suggestion or question, feel free to leave a message here or contact Dongjie Ye (dj.ye@my.cityu.edu.hk).
   
