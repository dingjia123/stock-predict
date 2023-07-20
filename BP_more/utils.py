import torch
from torch import nn
from torch.autograd import Variable


def get_rmse_log(model, feature, label, use_gpu):
    model.eval()
    mse_loss = nn.MSELoss()
    if use_gpu:
        feature = feature.cuda()
        label = label.cuda()
    feature = Variable(feature, volatile=True)
    label = Variable(label, volatile=True)
    pred = model(feature)
    clipped_pred = torch.clamp(pred, 1, float('inf'))
    rmse = torch.sqrt(mse_loss(clipped_pred.log(), label.log()))
    return rmse.item()


def get_mse_log(model, feature, label, use_gpu):
    model.eval()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        if use_gpu:
            feature = feature.cuda()
            label = label.cuda()
        else:
            feature = feature
            label = label
    pred = model(feature)
    if len(label.shape)==1:
        label = label.reshape(len(label),1)
    mse = mse_loss(pred, label)
    return mse.item()


def get_cross_log(model, feature, label, use_gpu):
    model.eval()
    cross_loss = nn.CrossEntropyLoss()
    if use_gpu:
        feature = feature.cuda()
        label = label.cuda()
    feature = Variable(feature, volatile=True)
    label = Variable(label, volatile=True)
    pred = model(feature)
    loss = cross_loss(pred, label)
    return loss.item()

# loss = nn.CrossEntropyLoss()
# Y = torch.tensor([2,0,1])       #三个目标值
# Y_pred_good = torch.tensor(       #三组待预测
#     [[0.1, 0.2, 3.9],
#     [1.2, 0.1, 0.3],
#     [0.3, 2.2, 0.2]])
#
# l1 = loss(Y_pred_good, Y)
# print(f'Batch Loss1: {l1.item():.4f}')
# print(Y_pred_good.dtype)
# print(Y.dtype)
# print(Y_pred_good)
# print(Y)



# mse_loss = nn.MSELoss(reduce=True,size_average=True)
# label = torch.tensor([[1.0],[8]])
# pred = torch.tensor([[3.0],[4]])
# print(pred.shape)
# mse = mse_loss(pred, label)
# print(mse)


