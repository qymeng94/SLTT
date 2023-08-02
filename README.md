# SLTT
This repository is the official PyTorch implementation of paper: Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks (ICCV2023).

## Dependencies
- Python 3
- PyTorch, torchvision
- spikingjelly 0.0.0.0.12
- Python packages: `pip install tqdm progress torchtoolbox`


## Training
We use one single V100 or A100 GPU for running all the experiments. Multi-GPU training is not supported in the current codes.
### SLTT
CIFAR-10, CIFAR-100, DVS-CIFAR10, and DVS-Gesture:

    # CIFAR-10
	python train_SLTT.py -data_dir ./data -dataset cifar10 -model spiking_resnet18 -amp -T_max 200 -epochs 200 -weight_decay 5e-5
    
    # CIFAR-100
    python train_SLTT.py -data_dir ./data -dataset cifar100 -model spiking_resnet18 -amp -T_max 200 -epochs 200 -weight_decay 5e-4
       
    # DVS-CIFAR10
	python train_SLTT.py -data_dir ./data/CIFAR10DVS -dataset DVSCIFAR10 -T 10 -amp -drop_rate 0.3 -model spiking_vgg11_bn -lr=0.05 -weight_decay 5e-4 -mse_n_reg
	
	# DVS-Gesture
    python train_SLTT.py -data_dir ./data/dvsgesture -dataset dvsgesture -model spiking_vgg11_bn -T 20 -b 16 -amp -drop_rate 0.4 -weight_decay 5e-4


ImageNet:
    
    ### Stage 1: train the models with T=1. 
    # The following three commands correspond to ResNet-34, ResNet-50, and ResNet-101, respectively.
	python train_SLTT.py -data_dir ./data/imagenet/ -dataset imagenet -T 1 -amp -epochs 100 -b 256 -T_max 100 -weight_decay 1e-5 -loss_lambda 0.0 -model spiking_nfresnet34
	python train_SLTT.py -data_dir ./data/imagenet/ -dataset imagenet -T 1 -amp -epochs 100 -b 256 -T_max 100 -weight_decay 1e-5 -loss_lambda 0.0 -model spiking_nfresnet50
	python train_SLTT.py -data_dir ./data/imagenet/ -dataset imagenet -T 1 -amp -epochs 100 -b 256 -T_max 100 -weight_decay 1e-5 -loss_lambda 0.0 -model spiking_nfresnet101
	
	### Rename the stage-1 checkpoints to "imagenet_resnet34_t1.pth", "imagenet_resnet50_t1.pth", and "imagenet_resnet101_t1.pth", respectively.
	### Move the renamed checkpoints to ./logs
	
    ### Stage 2: Fine-tune the stage-1 models with T=6 and fewer epochs.
	python train_SLTT.py -data_dir ./data/imagenet/ -dataset imagenet -T 6 -amp -epochs 30 -b 256 -T_max 30 -weight_decay 0.0 -loss_lambda 0.0 -model spiking_nfresnet34 -lr 0.001 -pre_train ./logs/imagenet_resnet34_t1.pth
	python train_SLTT.py -data_dir ./data/imagenet/ -dataset imagenet -T 6 -amp -epochs 30 -b 256 -T_max 30 -weight_decay 0.0 -loss_lambda 0.0 -model spiking_nfresnet50 -lr 0.001 -pre_train ./logs/imagenet_resnet50_t1.pth
	python train_SLTT.py -data_dir ./data/imagenet/ -dataset imagenet -T 6 -amp -epochs 30 -b 256 -T_max 30 -weight_decay 0.0 -loss_lambda 0.0 -model spiking_nfresnet101 -lr 0.001 -pre_train ./logs/imagenet_resnet101_t1.pth
    
The stage-1 models for ImageNet can be downloaded [here](https://cuhko365-my.sharepoint.com/:f:/g/personal/219019044_link_cuhk_edu_cn/EmxS-tKuDFlHlUV0UqM7CbQB8bdHy5Hvy_clwliBt6Pv4w?e=LHwcRI).

### SLTT-K
For DVS-Gesture, DVS-CIFAR10, and ImageNet, please run the following example codes:

    # DVS-Gesture
    python train_SLTTK.py -K 4 -data_dir ./data/dvsgesture -dataset dvsgesture -model spiking_vgg11_bn -T 20 -b 16 -amp -drop_rate 0.4 -weight_decay 5e-4
    
    # DVS-CIFAR10
	python train_SLTTK.py -K 2 -data_dir ./data/CIFAR10DVS/ -dataset DVSCIFAR10 -T 10 -amp -drop_rate 0.3 -model spiking_vgg11_bn -lr=0.05 -weight_decay 5e-4 -mse_n_reg

    # ImageNet
	python train_SLTTK.py -K 2 -data_dir ./data/imagenet/ -dataset imagenet -T 6 -amp -epochs 30 -b 256 -T_max 30 -weight_decay 0.0 -loss_lambda 0.0 -model spiking_nfresnet101 -lr 0.001 -pre_train ./logs/imagenet_resnet101_t1.pth

### BPTT
We also provide the BPTT implementation for comparison. For running the BPTT method, please refer directly to the commands for SLTT while changing "train_SLTT.py" to "train_BPTT.py". 
We give an example for CIFAR-10 as following.

	python train_BPTT.py -data_dir ./data -dataset cifar10 -model spiking_resnet18 -amp -T_max 200 -epochs 200 -weight_decay 5e-5
    

## Credits

The code for data preprocessing and neuron models is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo. The code for some utils is from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repo.


