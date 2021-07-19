import os
import argparse
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.chiebot_hxq_data import HxqDatasetLoader
from loss import OneClassLoss
from model import FCDDClassifier

import debug_tools as D


def parse_args():
    parser=argparse.ArgumentParser(description="train args set")
    parser.add_argument("--checkpoint",type=str,default="/home/chiebotgpuhq/MyCode/python/pytorch/FCDD/fcdd/data/results/fcdd_20210629133349_mvtec_/normal_0/it_0/snapshot.pt",help='')
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--data_dir",type=str,default="/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/ex_dataset/")
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--save_dir",type=str,default="/home/chiebotgpuhq/MyCode/python/pytorch/fcdd_classify/weight")
    args=parser.parse_args()
    return args

def train_one_loop(dataloader,model,loss_fn,optimizer):
    show_loss=0
    for batch_idx,(data,label,gt) in enumerate(dataloader):
        data=data.cuda()
        label=label.cuda().to(torch.float32)
        pred,x_f=model(data)
        pred=pred.squeeze()
        # breakpoint()
        loss=loss_fn(pred,label)
        x_f=x_f.detach().cpu().numpy()
        x_f=np.mean(x_f,axis=1)
        show_data=[x for x in x_f]
        D.show_img(data)
        D.show_img(show_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        show_loss+=loss.item()
        if batch_idx %100 == 0:
            loss=loss.item()
            print("当前损失是: {}  ".format(loss))
    return show_loss/len(dataloader)

def test_one_loop(dataloader,model,loss_fn):
    loss=0
    pred_colleter=[]
    label_colleter=[]
    with torch.no_grad():
        for batch_idx,(data,label,gt) in enumerate(dataloader):
            data=data.cuda()
            label=label.cuda()
            pred=model(data)
            pred=pred.squeeze()
            pred_colleter.append(pred.detach().cpu().numpy())
            label_colleter.append(label.squeeze().detach().cpu().numpy())
            loss+=loss_fn(pred,label).item()
    
    avg_loss=loss/len(dataloader.dataset)
    print("avg test loss is: {}".format(avg_loss))
    return avg_loss,evaluate_test(pred_colleter,label_colleter)

def evaluate_test(pre_colleter,label_colleter):
    if isinstance(pre_colleter,list):
        pre=np.hstack(pre_colleter)
        pre_c=(pre>0.5).astype(np.int) # type: ignore
    if isinstance(label_colleter,list):
        label=np.hstack(label_colleter)
    return label[label==pre_c].sum()/label.size


def main():
    args=parse_args()
    dataset=HxqDatasetLoader(args.data_dir)
    train_dataloader=DataLoader(dataset.train_set,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=2)
    test_dataloader=DataLoader(dataset.test_set,batch_size=args.batch_size,shuffle=False,drop_last=True,num_workers=2)
    current_time=datetime.datetime.now().strftime('%Y%m%d_%H_%M')
    model_save_dir=os.path.join(args.save_dir,current_time)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_save_path=os.path.join(model_save_dir,'best_model.ckpt')
    
    writer=SummaryWriter()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=FCDDClassifier().to(device)
    if os.path.exists(args.checkpoint):
        model_param=torch.load(args.checkpoint)
        model.load_state_dict(model_param['net'],strict=False)
        model.features.requires_grad=False
    # loss_fn=OneClassLoss()
    loss_fn=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    sche=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)


    current_test_loss=1000000
    for epoch in range(args.epochs):
        print("current epoch is {}".format(epoch))
        epoch_train_loss=train_one_loop(train_dataloader,model,loss_fn,optimizer)
        epoch_test_loss,right_rate=test_one_loop(test_dataloader,model,loss_fn)
        if current_test_loss> epoch_test_loss:
            torch.save(model.state_dict(),model_save_path)
        sche.step(epoch)
        print("current lr: {}".format(optimizer.param_groups[0]['lr'])) # type: ignore
        print(" 当前正确率:{}".format(right_rate))
        writer.add_scalar('train loss', epoch_train_loss,epoch)
        writer.add_scalar('test loss', epoch_test_loss,epoch)
        writer.add_scalar('lr',optimizer.param_groups[0]['lr'],epoch) # type: ignore
        writer.add_scalar('right rate', right_rate,epoch) # type: ignore

    print("train done")



if __name__=="__main__":
    main()


