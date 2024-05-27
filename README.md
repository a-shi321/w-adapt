# w-adapt




# Preparation
## Install Pytorch

 Our code is conducted based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch),please setup the framework by it.

## Download dataset

we use pascal voc and watercolor datasets respectly as source and target,the pascal voc dataset could be download [Here](http://host.robots.ox.ac.uk/pascal/VOC/)
and the watercolor dataset could be download [Here](https://naoto0804.github.io/cross_domain_detection/)

the format of datasets is similar with VOC,you just need to split train.txt to train_s.txt and train_t.txt


## Train and Test
1.train the model,you need to download the pretrained model [resnet-101](https://github.com/jwyang/faster-rcnn.pytorch) which is different with pure pytorch pretrained model

2.change the dataset root path in ./lib/model/utils/config.py and some dataset dir path in ./lib/datasets/cityscape.py,the default data path is ./data

3 Train the model

## train pascal voc -> watercolor
CUDA_VISIBLE_DEVICES=GPU_ID python da_trainval_net.py --dataset VOC2water --net res101 --bs 1 --lr 2e-3 --lr_decay_step 6 --cuda

## Test model in target domain 
CUDA_VISIBLE_DEVICES=GPU_ID python test.py --dataset cityscape --part test_t --model_dir=# The path of your pth model --cuda
