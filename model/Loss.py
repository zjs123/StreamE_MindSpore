# import torch
# import torch.autograd as autograd
# import torch.nn.functional as F
# import torch.nn as nn

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import Zero

# USE_CUDA = torch.cuda.is_available()
# if USE_CUDA:
# 	longTensor = torch.cuda.LongTensor
# 	floatTensor = torch.cuda.FloatTensor
#
# else:
# 	longTensor = torch.LongTensor
# 	floatTensor = torch.FloatTensor


class StableBCELoss(nn.Cell):
    def __init__(self):
        super(StableBCELoss, self).__init__()
    def construct(self, input, target):
        neg_abs = - input.abs()
        loss = ms.ops.clip_by_value(input, clip_value_min=0) - input * target + (1 + neg_abs.exp()).log()
        # loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        # loss = -(target*input.log() + (1-target)*(1-input).log())
        return loss.mean()

# class StableBCELoss(nn.modules.Module):
#     def __init__(self):
#         super(StableBCELoss, self).__init__()
#
#     def forward(self, input, target):
#         neg_abs = - input.abs()
#         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
#         #loss = -(target*input.log() + (1-target)*(1-input).log())
#         return loss.mean()

class sigmoidLoss(nn.Cell):
    def __init__(self):
        super(sigmoidLoss, self).__init__()
        self.logsigmoid = ms.nn.LogSigmoid()

    def construct(self, pos, neg):
        pos_score = -self.logsigmoid(pos).mean()
        neg_score = -self.logsigmoid(-neg).mean()
        return (pos_score + neg_score) / 2

# class sigmoidLoss(nn.Module):
#     def __init__(self):
#         super(sigmoidLoss, self).__init__()
#
#     def forward(self, pos, neg):
#         pos_score = -F.logsigmoid(pos).mean()
#         neg_score = -F.logsigmoid(-neg).mean()
#         return (pos_score+neg_score)/2

# class marginLoss(nn.Module):
# 	def __init__(self):
# 		super(marginLoss, self).__init__()
# 	# def forward(self, pos, neg, margin):
# 	def forward(self, pos, neg, margin):
# 		zero_tensor = floatTensor(pos.size())
# 		zero_tensor.zero_()
# 		zero_tensor = autograd.Variable(zero_tensor)
# 		return torch.mean(torch.max(pos - neg + margin, zero_tensor))

class marginLoss(nn.Cell):
    def __init__(self):
        super(marginLoss, self).__init__()

    def construct(self, pos, neg, margin):
        zero_tensor = ms.Tensor(shape=pos.shape, dtype=ms.float32, init=Zero())
        return ms.ops.mean(ms.ops.maximum(pos - neg + margin, zero_tensor))

# class marginLoss(nn.Module):
#     def __init__(self):
#         super(marginLoss, self).__init__()
#
#     def forward(self, pos, neg, margin):
#         zero_tensor = floatTensor(pos.size())
#         zero_tensor.zero_()
#         zero_tensor = autograd.Variable(zero_tensor)
#         return torch.mean(torch.max(pos - neg + margin, zero_tensor))
    
class double_marginLoss(nn.Cell):
    def __init__(self):
        super(double_marginLoss, self).__init__()

    def construct(self, pos, neg, margin):
        zero_tensor = ms.Tensor(shape=pos.shape, dtype=ms.float32, init=Zero())
        # zero_tensor.zero_()

        pos_margin = 1.0
        neg_margin = pos_margin + margin
        neg_max= ms.ops.maximum(neg_margin - neg, zero_tensor)
        pos_max= ms.ops.maximum(pos - pos_margin, zero_tensor)
        return ms.ops.sum(neg_max) + ms.ops.sum(pos_max)


# class double_marginLoss(nn.Module):
#     def __init__(self):
#         super(double_marginLoss,self).__init__()
#
#     def forward(self, pos, neg, margin):
#         zero_tensor = floatTensor(pos.size())
#         zero_tensor.zero_()
#         zero_tensor = autograd.Variable(zero_tensor)
#
#         pos_margin=1.0
#         neg_margin=pos_margin+margin
#         return torch.sum(torch.max(neg_margin-neg, zero_tensor))+torch.sum(torch.max(pos-pos_margin,zero_tensor))

def orthogonalLoss(rel_embeddings, norm_embeddings):
    return ms.ops.sum(ms.ops.sum(ms.ops.mm(rel_embeddings,norm_embeddings), dim=1, keepdim=True) ** 2 / ms.ops.sum(rel_embeddings ** 2, dim=1, keepdim=True))
	# return torch.sum(torch.sum(torch.mm(rel_embeddings,norm_embeddings), dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def normLoss(embeddings, dim=1):

    norm = ms.ops.sum(embeddings ** 2, dim=dim, keepdim=True)

    # norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
    max_temp= ms.ops.maximum(norm - ms.Tensor([1.0], dtype=ms.float32), ms.Tensor([0.0], ms.float32))
    return ms.ops.mean(max_temp)
	# return torch.mean(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))

def F_norm(martix):

    norm = ms.ops.norm(martix)
    # norm = torch.norm(martix)
    max_res, _ = ms.ops.maximum(norm - ms.Tensor([1.0], ms.float32), ms.Tensor([0.0], ms.float32))
    return max_res
    # return torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0])))
