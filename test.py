import torch
from torch.utils.data import DataLoader
import numpy as np

from data.chiebot_hxq_data import HxqDatasetLoader
from model import FCDDClassifier
import debug_tools as D

def main():
    dataset=HxqDatasetLoader("/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/ex_dataset/")
    test_dataloader=DataLoader(dataset.test_set,batch_size=4,shuffle=False,drop_last=True,num_workers=2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=FCDDClassifier().to(device)
    model_param=torch.load("/home/chiebotgpuhq/MyCode/python/pytorch/fcdd_classify/weight/20210719_12_43/best_model.ckpt")
    model.load_state_dict(model_param,strict=False)
    model.eval()
    with torch.no_grad():
        for batch_idx,(data,label,gt) in enumerate(test_dataloader):
            data=data.cuda()
            label=label.cuda()
            pred,x_f=model(data)
            pred=pred.squeeze()
            print(label)
            print(pred)
            D.show_img(data)
            x_f=x_f.detach().cpu().numpy()
            x_f=np.mean(x_f,axis=1)
            show_data=[x for x in x_f]
            D.show_img(show_data)


if __name__=="__main__":
    main()
