# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 07月 15日 星期四 15:58:42 CST
@Description: 
'''
import torch
import torch.nn as nn

class OneClassLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,out:torch.Tensor,label:torch.Tensor):
        loss=(out**2+1).sqrt() -1
        norm=loss[label==0]
        anorm=loss[label==1]

        anorm_loss=-torch.log(1-((-anorm).exp())+1e-31) # type: ignore
        loss[label==1]=anorm_loss
        return  loss.mean()




