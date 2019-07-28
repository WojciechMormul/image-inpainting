PyTorch implementation of paper: Globally and Locally Consistent Image Completion.

install pytorch: pip install torch==0.4.1
install torchvision: pip install torchvision
install opencv: pip install opencv-python
install roialign: https://github.com/longcw/RoIAlign.pytorch/tree/pytorch_0.4

CUDA_VISIBLE_DEVICES=0,1 python train.py
CUDA_VISIBLE_DEVICES=0 python eval.py
