import time
import shutil
import sys
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from CandyNet import *
import MyDataloader
import SimpleITK as sitk
import pandas as pd
import torchvision
from visdom import Visdom
from medpy import metric
from utils import *
import numpy as np
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../')
viz = Visdom(env='Server3Net')
# viz.line([0], [0], win='train loss', opts=dict(title='train loss',showlegen = True))
# viz.line([0], [0], win='valid loss', opts=dict(title='valid loss',showlegen = True))
# viz.line([0], [0], win='train avg dice', opts=dict(title='train avg dice'))
# viz.line([0], [0], win='valid avg dice', opts=dict(title='valid avg dice'))
viz.line([0], [0], win='train avg loss', opts=dict(title='train avg loss'))
viz.line([0], [0], win='valid avg loss', opts=dict(title='valid avg loss'))

#################initialization network##############
def weights_init(model):
	if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
		nn.init.kaiming_uniform_(model.weight.data, 0.25)
		nn.init.constant_(model.bias.data, 0)
	# elif isinstance(model, nn.InstanceNorm3d):
	# 	nn.init.constant_(model.weight.data,1.0)
	# 	nn.init.constant_(model.bias.data, 0)

def train_valid_seg():
	#####################paras & config###########################################
	if_test = False
	if_resume = True
	max_epoches = 100
	saved_rounds = 1
	train_batch_size = 1
	valid_batch_size =1;test_batch_size = 1
	channels=1
	depth=16
	height=256
	width=256
	learning_rate = 0.01
	weight_decay = 1e-4

	config = {
		# 'model':'USNETres',
		'train_csv': './train.csv',
		'valid_csv': './valid.csv',
		'test_csv': './test.csv',
		'ckpt_dir': './results/',
		'saved_dir':"/home/lihuiyu/Data/LiTS/segResults/",
		'model_dir' : "./ModelAdd/PiaNet_GRU.pth"
	}
	#####################paras & config##########################################
	# refresh save dir
	exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
	ckpt_dir = os.path.join(config['ckpt_dir'] + exp_id)
	saved_dir = os.path.join(config['saved_dir'] + exp_id)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(saved_dir):
		os.makedirs(saved_dir)
	# # Clear新数据存储路径
	# if os.path.exists(config['saved_dir']) is True:
	# 	shutil.rmtree(config['saved_dir'])
	# os.mkdir(config['saved_dir'])

	logfile = os.path.join(ckpt_dir, 'log')
	if if_test != 1:
		sys.stdout = Logger(logfile)

	dicelog = 'dice.txt'
	if os.path.isfile(dicelog):
		os.remove(dicelog)
	else:
		with open(dicelog, 'w') as log:
			log.write(str(datetime.now()) + '\n')
	###############GPU set####################################
	torch.manual_seed(0)
	# torch.cuda.set_device(0)
	if torch.cuda.is_available():
		net = PiaNet().cuda()#need to do this before constructing optimizer
		loss = MTLloss().cuda()
	else:
		net = PiaNet()
		loss = MTLloss()
	cudnn.benchmark = False  # True
	# net = DataParallel(net).cuda()
	optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9,weight_decay=weight_decay)#SGD+Momentum
	# optimizer = torch.optim.Adam(net.parameters(), args.lr,(0.9, 0.999),eps=1e-08,weight_decay=2e-4)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#decay the learning rate after 100 epoches
	################resume or initialize prams #############################
	if if_test or if_resume:
		print('if_test:',if_test,'if_resume:',if_resume)
		checkpoint = torch.load(config['model_dir'])
		net.load_state_dict(checkpoint)
		# net.load_state_dict(torch.load('params.pkl'))
	else:
		print('weight initialization')
		net.apply(weights_init)

	#test
	if if_test:
		print('###################test###################')
		test_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['test_csv'],
								 channels=channels, depth=depth, height=height,width=width),
								 batch_size=test_batch_size, shuffle=False, pin_memory=True)
		test_loss, test_iter = test(test_loader, net, loss, saved_dir)
		test_avgloss = sum(test_loss) / test_iter
		print("test loss:%.4f, Time:%.3f min" % (test_avgloss, (time.time() - start_time) / 60))
		return

	# train_set_loader
	train_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['train_csv'],
							  channels=channels,depth=depth, height=height, width=width),
	                          batch_size=train_batch_size, shuffle=True,pin_memory=True)
	# print(len(train_loader))#

	# val_set_loader
	val_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['valid_csv'],
							channels=channels, depth=depth, height=height, width=width),
	                        batch_size=valid_batch_size, shuffle=False,pin_memory=True)
	#################train-eval (epoch)##############################
	min_validloss = 1.
	for epoch in range(max_epoches):
		####set optimizer lr#################
		print('###################train epoch',str(epoch),'lr=',str(optimizer.param_groups[0]['lr']),'###################')
		train_loss,train_iter = train(train_loader, net, loss, optimizer,epoch,saved_dir,saved_rounds)
		scheduler.step(epoch) #In PyTorch 1.1.0 and later,`optimizer.step()` before `lr_scheduler.step()`
		train_avgloss = sum(train_loss) / train_iter
		print("[%d/%d], train loss:%.4f, Time:%.3f min" %\
		      (epoch, max_epoches, train_avgloss, (time.time() - start_time) / 60))

		print('###################valid',str(epoch),'###################')
		valid_loss,valid_iter = validate(val_loader, net, loss,epoch,saved_dir,saved_rounds)
		valid_avgloss = sum(valid_loss) / valid_iter
		print("[%d/%d], train loss:%.4f, Time:%.3f min " % \
			  (epoch, max_epoches, valid_avgloss, (time.time() - start_time) / 60))
		# print:epoch/total,loss123,lr,accurate,time

		# viz.line(train_loss,np.linspace(0,train_iter,train_iter), win='train loss',update='append')
		# viz.line(valid_loss,np.linspace(0,valid_iter,valid_iter), win='valid loss',update='append')
		# viz.line([train_dice1], [epoch], win='train avg dice', update='append')
		# viz.line([valid_dice1], [epoch], win='valid avg dice', update='append')
		viz.line([train_avgloss], [epoch], win='train avg loss', update='append')
		viz.line([valid_avgloss], [epoch], win='valid avg loss', update='append')

		#if-save:
		if min_validloss > abs(valid_avgloss):
			min_validloss = abs(valid_avgloss)
			# print('abs(valid_loss)',abs(valid_avgloss))
		# if epoch %10 is 0:
			state = {
				'epoche':epoch,
				'arch':str(net),
				'state_dict':net.state_dict(),
				'optimizer':optimizer.state_dict()
				#other measures
			}
			torch.save(state,ckpt_dir+'/checkpoint.pth.tar')
			#save model
			model_filename = ckpt_dir+'/model_'+str(epoch)+'.pth'
			torch.save(net.state_dict(),model_filename)
			print('Model saved in',model_filename)

def train(data_loader, net, loss, optimizer,epoch,saved_dir,saved_rounds):
	net.train()#swithch to train mode
	epoch_loss = []
	total_iter = len(data_loader)
	for i, (data,target,origin,direction,space,ct_name) in enumerate(data_loader):
		if torch.cuda.is_available():
			data = data.cuda()
			target = target.cuda()
		output1, output2 = net(data)
		loss_output = loss(output1, output2, target)
		tumor_dice, liver_dice = Dice(output1, output2, target)
		optimizer.zero_grad()#set the grade to zero
		loss_output.backward()
		optimizer.step()
		epoch_loss.append(loss_output.item())#Use tensor.item() to convert a 0-dim tensor to a Python number
		print("[%d/%d], loss:%.4f, liver_dice:%.4f, tumor_dice:%.4f" % (i, total_iter, loss_output.item(),liver_dice,tumor_dice))
		# viz.line([loss_output.item()], [i], win ='train loss', update='append',opts=dict(title='train loss'))
		# if epoch % saved_rounds == 0:
		# 	output1_name1 = os.path.join(saved_dir, 'valid-' + ct_name[0] + '-' + str(epoch) + '-output1' + '.nii')
		# 	output2_name2 = os.path.join(saved_dir, 'valid-' + ct_name[0] + '-' + str(epoch) + '-output2' + '.nii')
		# 	origin = tuple(k.item() for k in origin)
		# 	direction = tuple(k.item() for k in direction)
		# 	space = tuple(k.item() for k in space)
		# 	saved_preprocessed(output1, origin, direction, space, output1_name1)
		# 	saved_preprocessed(output2, origin, direction, space, output2_name2)
	return epoch_loss,total_iter

def validate(data_loader, net, loss,epoch,saved_dir,saved_rounds):
	net.eval()
	epoch_loss = []
	total_iter = len(data_loader)
	with torch.no_grad():#no backward
		for i, (data,target,origin,direction,space,ct_name) in enumerate(data_loader):
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			output1,output2 = net(data)
			loss_output = loss(output1,output2,target)
			tumor_dice, liver_dice = Dice(output1, output2, target)
			epoch_loss.append(loss_output.item())#Use tensor.item() to convert a 0-dim tensor to a Python number
			print("[%d/%d], loss:%.4f, liver_dice:%.4f, tumor_dice:%.4f" % (i, total_iter, loss_output.item(), liver_dice, tumor_dice))
			# viz.line([loss_output.item()], [i], win='valid loss', update='append', opts=dict(title='valid loss'))
			if epoch%saved_rounds == 0:
				output1_name1 = os.path.join(saved_dir, 'valid-' + ct_name[0] + '-' + str(epoch) + '-output1' + '.nii')
				output2_name2 = os.path.join(saved_dir, 'valid-' + ct_name[0] + '-' + str(epoch) + '-output2' + '.nii')
				origin = tuple(k.item() for k in origin)
				direction = tuple(k.item() for k in direction)
				space = tuple(k.item() for k in space)
				saved_preprocessed(output1, origin, direction, space, output1_name1)
				saved_preprocessed(output2, origin, direction, space, output2_name2)


	return epoch_loss, total_iter

def test(data_loader, net, loss, saved_dir):
	net.eval()
	epoch_loss = []
	total_iter = len(data_loader)
	with torch.no_grad():  # no backward
		for i, (data, target, origin, direction, space, ct_name) in enumerate(data_loader):
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			output1, output2 = net(data)
			loss_output = loss(output1, output2, target)
			tumor_dice, liver_dice = Dice(output1, output2, target)
			epoch_loss.append(loss_output.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
			print("[%d/%d], loss:%.4f, liver_dice:%.4f, tumor_dice:%.4f" % (
			i, total_iter, loss_output.item(), liver_dice, tumor_dice))
			viz.line([loss_output.item()], [i], win='valid loss', update='append', opts=dict(title='valid loss'))
			#saved as nii
			output1_name1 = os.path.join(saved_dir, 'valid-' + ct_name[0] + '-output1' + '.nii')
			output2_name2 = os.path.join(saved_dir, 'valid-' + ct_name[0] + '-output2' + '.nii')
			origin = tuple(k.item() for k in origin)
			direction = tuple(k.item() for k in direction)
			space = tuple(k.item() for k in space)
			saved_preprocessed(output1, origin, direction, space, output1_name1)
			saved_preprocessed(output2, origin, direction, space, output2_name2)

	return epoch_loss, total_iter

if __name__ == '__main__':
	# print(torch.__version__)#0.4.1
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
	start_time = time.time()
	train_valid_seg()
	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))