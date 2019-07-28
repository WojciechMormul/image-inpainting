PyTorch implementation of paper: Satoshi Iizuka: Globally and Locally Consistent Image Completion.

install pytorch: pip install torch==0.4.1</br>
install opencv: pip install opencv-python</br>
install roialign: https://github.com/longcw/RoIAlign.pytorch/tree/pytorch_0.4</br>

CUDA_VISIBLE_DEVICES=0,1 python train.py</br>
CUDA_VISIBLE_DEVICES=0 python eval.py
