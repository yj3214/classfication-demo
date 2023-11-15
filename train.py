import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from monai.transforms import LoadImage, ToTensor,Compose,Resize,EnsureChannelFirst,NormalizeIntensity
from monai.data import NibabelReader,DataLoader
from monai.networks.nets import DenseNet121
from utils import read_split_data,MyDataset,train_one_epoch,evaluate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
def main(args):
    train_data,val_data = read_split_data(args.data_root,val_rate=args.val_rate)
    # 定义数据转换
    transform = Compose([
        LoadImage(reader=NibabelReader,image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=(256,256,256),mode='trilinear'),
        ToTensor(),
        NormalizeIntensity(channel_wise=True)
    ])
    train_dataset = MyDataset(train_data,transform=transform)
    val_dataset = MyDataset(val_data,transform=transform)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=0)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=0)
    model = DenseNet121(spatial_dims=3,in_channels=1,out_channels=args.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learn_rate, momentum=0.99, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        # train
        train_one_epoch(model=model,optimizer=optimizer,data_loader=train_loader,device=device,epoch=epoch)
        scheduler.step()
        if (epoch+1) % 10 == 0:
            # validate
            evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
            torch.save(model.state_dict(), args.save_path+'/train_epoch_'+str(epoch+1)+'.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据的路径
    parser.add_argument('--data_root', type=str, default='./data', help='Description of data_root')
    parser.add_argument('--val_rate', type=float, default=0.5, help='Description of val_rate')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Description of train learn_rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Description of train weight_decay')
    parser.add_argument('--epochs', type=int, default=50, help='Description of train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Description of train batch_size')
    parser.add_argument('--num_classes', type=int, default=2, help='Description of num_classes')
    # 提前创建好保存权重的路径
    parser.add_argument('--save_path', type=str, default='./weight', help='Description of num_classes')
    args = parser.parse_args()
    main(args)