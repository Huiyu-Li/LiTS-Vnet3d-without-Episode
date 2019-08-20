import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os
import shutil
from torch import nn
from MyDataLoader import LSDataloader
from torch.utils.data import DataLoader
from MyGRU import GRUModel,diceLoss,SimpleNet,ceLoss
from visdom import Visdom
import numpy as np
viz = Visdom(env='MyGRUq')
viz.line([0], [0], win='train loss', opts=dict(title='train loss'))
viz.line([0], [0], win='valid loss', opts=dict(title='valid loss'))
viz.line([0], [0], win='train avg loss', opts=dict(title='train avg loss'))
viz.line([0], [0], win='valid avg loss', opts=dict(title='valid avg loss'))

def weights_init(model):
    if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
        nn.init.xavier_normal_(model.weight.data, 0.25)
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm3d)or isinstance(model, nn.InstanceNorm3d):
        nn.init.constant_(model.weight.data,1.0)
        nn.init.constant_(model.bias.data, 0)

def main():
    max_epochs = 100
    num_layers = 20
    model_dir = ''
    if os.path.isdir('./MCVSeg/'):
        shutil.rmtree('./MCVSeg/')
    os.mkdir('./MCVSeg/')

    if torch.cuda.is_available():
        net = GRUModel(num_layers).cuda()
        loss = diceLoss().cuda()
    else:
        net = GRUModel(num_layers)#GRUModel num_layers
        loss = diceLoss()
        # loss = ceLoss()

    if model_dir:
        print('weight resume')
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint)
    else:
        print('weight initialization')
        net.apply(weights_init)

    # train_set_loader
    train_loader = DataLoader(LSDataloader('./train.csv'),batch_size=1, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(LSDataloader('./valid.csv'),batch_size=1, shuffle=True, pin_memory=True)

    #######################network define#################
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)  # SGD+Momentum
    # optimizer = torch.optim.Adam(net.parameters(), args.lr,(0.9, 0.999),eps=1e-08,weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # decay the learning rate after 100 epoches
    for epoch in range(max_epochs + 1):
        ####set optimizer lr#################
        scheduler.step(epoch)  # must before optimizer.step
        print('####train epoch', str(epoch), '####')
        epoch_loss, total_epoch, output,target = train(train_loader, net, loss, optimizer, epoch)
        train_avgloss = sum(epoch_loss) / total_epoch
        print("[%d/%d], train loss:%.4f, Time:%.3f min" % (epoch, max_epochs + 1, train_avgloss,(time.time() - start_time) / 60))
        viz.line([train_avgloss], [epoch], win='train avg loss', update='append')

        print('####valid epoch', str(epoch), '####')
        epoch_loss, total_epoch, output = valid(valid_loader, net, loss, epoch)
        valid_avgloss = sum(epoch_loss) / total_epoch
        print("[%d/%d], valid loss:%.4f, Time:%.3f min" % (epoch, max_epochs + 1, valid_avgloss, (time.time() - start_time) / 60))
        viz.line([valid_avgloss], [epoch], win='valid avg loss', update='append')

        if epoch % 10 is 0:
            state = {
                'epoche': epoch,
                'arch': str(net),
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
                # other measures
            }
            torch.save(state, './ckptDir/checkpoint.pth.tar')
            # save model
            model_filename = './ckptDir/model_' + str(epoch) + '.pth'
            torch.save(net.state_dict(), model_filename)
            print('Model saved in', model_filename)


def train(data_loader, net, loss, optimizer, epoch):
    net.train()  # swithch to train mode
    epoch_loss = []
    total_epoch = len(data_loader)
    for i, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = Variable(data.cuda())
            target = Variable(target.cuda())
        else:
            data = Variable(data)
            target = Variable(target)
        output = net(data,data)  # run the model
        loss_output = loss(output, target)
        # import pdb
        # pdb.set_trace()
        # for name,param in net.named_parameters():
        #     # if item[0] == 'fc.2.fc.weight':
        #     print(name,param.sum())
        #     # h = param[1].register_hook(lambda grad: print(grad))

        optimizer.zero_grad()  # set the grade to zero
        loss_output.backward()
        optimizer.step()
        epoch_loss.append(loss_output.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
        print("[%d/%d], loss:%.4f" % (i, total_epoch, loss_output.item()))
        viz.line([loss_output.item()], [i], win ='train loss', update='append')
        if epoch % 100 is 0:
            fig = plt.figure()
            plt.subplot(131)
            plt.imshow(data[0, 0, 0, :, :].detach().cpu().numpy(),cmap='gray');plt.title('input')
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(target[0, 0, 0, :, :].detach().cpu().numpy(),cmap='gray');plt.title('target')
            plt.axis('off')
            plt.subplot(133)
            temp = np.argmax(output.detach().cpu().numpy(),1)
            plt.imshow(temp[0, 0, :, :],cmap='gray');plt.title('output')
            plt.axis('off')
            fig.tight_layout()
            plt.savefig('./MCVSeg/train' + str(epoch)+'_' + str(i)+ '.png')
            # plt.show(block=False)


    return epoch_loss, total_epoch,output,target

def valid(data_loader, net, loss, epoch):
    net.eval()
    epoch_loss = []
    total_epoch = len(data_loader)
    with torch.no_grad():  # no backward
        for i, (data, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                data = Variable(data.cuda())
                target = Variable(target.cuda())
            else:
                data = Variable(data)
                data = data*255
                target = Variable(target)
            output = net(data,data)
            loss_output = loss(output, target)  # do we need to split the target?
            epoch_loss.append(loss_output.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
            print("[%d/%d], loss:%.4f" % (i, total_epoch, loss_output.item()))
            viz.line([loss_output.item()], [i], win='valid loss', update='append')

            if epoch % 100 is 0:
                fig = plt.figure()
                plt.subplot(131)
                plt.imshow(data[0, 0, 0, :, :].detach().cpu().numpy(), cmap='gray');
                plt.title('input')
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(target[0, 0, 0, :, :].detach().cpu().numpy(), cmap='gray');
                plt.title('target')
                plt.axis('off')
                plt.subplot(133)
                temp = np.argmax(output.detach().cpu().numpy(), 1)
                plt.imshow(temp[0, 0, :, :], cmap='gray');
                plt.title('output')
                plt.axis('off')
                fig.tight_layout()
                plt.savefig('./MCVSeg/valid' + str(epoch) + '_' + str(i) + '.png')
                # plt.show(block=False)

    return epoch_loss, total_epoch, output

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))