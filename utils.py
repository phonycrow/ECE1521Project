import numpy as np
from pathlib import Path
from PIL import Image
import time
import torch
import torch.nn as nn
import torchvision

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def epoch(mode, dataloader, net, optimizer, criterion, args):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].float().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)

        loss_avg += loss.item()*n_b
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            transformed_img = np.uint8((output[0].detach().cpu().numpy().transpose((1, 2, 0)) + 0.5)*256)
            Image.fromarray(transformed_img).save(args.save_path / dataloader.dataset.get_order(i_batch))

    loss_avg /= num_exp

    return loss_avg

def train(it_eval, net, trainloader, testloader, args):
    #print(args)
    net = net.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.MSELoss().to(args.device)

    start = time.time()
    for ep in range(Epoch+1):
        print(f"Epoch {ep}")
        loss_train = epoch('train', trainloader, net, optimizer, criterion, args)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    torch.save(net.state_dict(), args.save_path / "net.pt")
    time_train = time.time() - start
    loss_test = epoch('test', testloader, net, optimizer, criterion, args)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f test loss = %.6f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, loss_test))

    return net
