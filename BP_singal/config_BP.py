# -- coding: utf-8 --
# 可以调整的超参
batch_size = 128
epochs = 600
use_gpu = False
lr = 0.001
weight_decay = 8

TIMESTEPS = 15          #循环网络中训练序列的长度
judge = 'close'   # high最高价，low最低价，close收盘价，open开盘价
