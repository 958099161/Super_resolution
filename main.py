import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model1 import RCAN
from san import SAN
from dataset import vedio_data
from utils import AverageMeter
from tensorboardX import SummaryWriter
import math

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def set_lr(args, epoch, optimizer):
    lr = args.lr / (1 + 1 * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='RCAN')
    # parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, default='weight')
    # parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    parser.add_argument('--model_savepath', default='E:\lunwen\RCAN-pytorch-master\weight', help='dataset directory')
    parser.add_argument('--model_name', default='RCAN_epoch_5_26_-2-211.pth', help='model directory')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    # model = RCAN(opt).to(device)
    model = SAN(opt).cuda()
    criterion = nn.L1Loss()
    if False:
        model_path=opt.model_savepath+"\\"+opt.model_name
        model.load_state_dict(torch.load(model_path))
        print("model from :{}".format(model_path))
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = vedio_data(opt)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    writer = SummaryWriter(log_dir="./logs/", comment='loss')
    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()
        learning_rate = set_lr(opt, epoch, optimizer)
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{} learn_rate{}'.format(epoch + 1, opt.num_epochs, learning_rate))
            for i,(data) in enumerate(dataloader):
                inputs, labels ,name= data
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                writer.add_scalar('mse_loss', loss, epoch * len(dataloader) + i)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))
                # if i %200 ==0:
            torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_5_28_-2-2{}.pth'.format(opt.arch, epoch)))
