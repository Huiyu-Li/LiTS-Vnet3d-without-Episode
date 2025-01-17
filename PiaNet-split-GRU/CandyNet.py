import torch
from torch import nn
import numpy as np
import SimpleITK as sitk
from medpy import metric

def saved_preprocessed(savedImg,origin,direction,space,saved_name):
	origin = tuple(k.item() for k in origin)
	direction = tuple(k.item() for k in direction)
	space = tuple(k.item() for k in space)#xyz_thickness
	savedImg = np.squeeze(np.argmax(savedImg.detach().cpu().numpy(),1),0).astype(np.float32)
	newImg = sitk.GetImageFromArray(savedImg)
	newImg.SetOrigin(origin)
	newImg.SetDirection(direction)
	newImg.SetSpacing(space)
	sitk.WriteImage(newImg, saved_name)

def Dice(output1, output2, target):
	pred_liver = np.argmax(output1.detach().cpu().numpy(), axis=1)
	pred_lesion = np.argmax(output2.detach().cpu().numpy(), axis=1)

	target = np.squeeze(target.detach().cpu().numpy(), axis=1)
	true_liver = target >= 1
	true_lesion = target == 2

	# Compute per-case (per patient volume) dice.
	liver_prediction_exists = np.any(pred_liver == 1)
	if not np.any(pred_lesion) and not np.any(true_lesion):
		tumor_dice = 1.
		print('tumor_dice = 1')
	else:
		tumor_dice = metric.dc(pred_lesion, true_lesion)
	if liver_prediction_exists:
		liver_dice = metric.dc(pred_liver, true_liver)
	else:
		liver_dice = 0
		print('liver_dice = 0')
	return tumor_dice, liver_dice

def one_hot(output, label):
	_labels = torch.zeros_like(output)
	_labels.scatter_(dim=1, index=label.long(), value=1)#带下划线的函数是inplace,内部赋值，没有返回值
	#scatter_(input, dim, index, value)将value中数据根据index中的索引按照dim的方向填进input中。
	_labels.requires_grad = False
	return _labels

class MTLloss(nn.Module):
	def __init__(self):
		super().__init__()
		self.smooth = 1e-5

	def forward(self, output1, output2, target):
		label = target.clone()
		label[target>=1]=1
		target1 = one_hot(output1, label)
		intersection1 = 2. * (output1 * target1).sum()
		denominator1 = output1.sum() + target1.sum()
		dice1 = (intersection1 + self.smooth) / (denominator1 + self.smooth)

		label[target == 1] = 0
		target2 = one_hot(output2, label)
		intersection2 = 2. * (output2 * target2).sum()
		denominator2 = output2.sum() + target2.sum()
		dice2 = (intersection2 + self.smooth) / (denominator2 + self.smooth)

		dice = 1 - 0.5 * (dice1 + dice2)
		return dice

class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes, self).__init__()
        self.resBlock = nn.Sequential(
            nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm3d(n_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_out, n_out, kernel_size=3, padding=1),
            nn.InstanceNorm3d(n_out)
        )
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.InstanceNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.resBlock(x)
        out += residual
        out = self.relu(out)
        return out

class GRUCell(nn.Module):
    def __init__(self):
        super(GRUCell, self).__init__()
        self.conv3d = nn.Conv3d(2,2,3,stride=1,padding=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        reset_1 = self.tanh(self.conv3d(input)+self.conv3d(hidden))
        reset_2 = self.tanh(self.conv3d(input)+self.conv3d(hidden))

        u_1 = torch.mul(reset_1,self.conv3d(input))+torch.mul((1-reset_1),input)
        u_2 = torch.mul(reset_2,self.conv3d(u_1))+torch.mul((1-reset_1),u_1)
        u_3 = self.conv3d(u_2)
        u_3 = self.softmax(u_3)
        return u_3

class GRUModel(nn.Module):
    def __init__(self,num_layers=4):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.GRUCell = GRUCell()
        self.input = nn.Sequential(
			nn.Conv3d(1, 2, kernel_size=1),
			nn.InstanceNorm3d(2),
			nn.ReLU(inplace=True),
			# nn.Dropout3d(p = 0.3),
			nn.Conv3d(2, 2, kernel_size=1))

    def forward(self, input, hidden):
        input = self.input(input)
        for _ in range(self.num_layers):
            hidden = self.GRUCell(input, hidden)
        return hidden

class SubGRUNet(nn.Module):
	def __init__(self, num_layers=10):
		super(SubGRUNet, self).__init__()
		self.GRUmodel1 = GRUModel()
		self.GRUmodel2 = GRUModel()

	def forward(self, input, hidden1, hidden2):
		output1 = self.GRUmodel1(input, hidden1)
		output2 = self.GRUmodel2(input, hidden2)
		return output1, output2

class Decoder1(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_blocks_back = [3, 3, 2, 2]  # [5-2]
		self.nff = [1, 8, 16, 32, 64, 128]  # NumFeature_Forw[0-5]
		self.nfb = [64, 32, 16, 8, 2]  # NunFeaturn_Back[5-0]
		#deconv4-1,output
		self.deconv4 = nn.Sequential(
			nn.ConvTranspose3d(self.nff[5], self.nfb[0], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[0]),
			nn.ReLU(inplace=True))
		self.deconv3 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[0], self.nfb[1], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[1]),
			nn.ReLU(inplace=True))
		self.deconv2 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[1], self.nfb[2], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[2]),
			nn.ReLU(inplace=True))
		self.deconv1 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[2], self.nfb[3], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[3]),
			nn.ReLU(inplace=True))
		self.output = nn.Sequential(
			nn.Conv3d(self.nfb[3], self.nfb[3], kernel_size=1),
			nn.InstanceNorm3d(self.nfb[3]),
		    nn.ReLU(inplace=True),
			# nn.Dropout3d(p = 0.3),
		    nn.Conv3d(self.nfb[3], self.nfb[4], kernel_size=1))  # since class number = 3 and split into 2 branch

		#backward4-1
		for i in range(len(self.num_blocks_back)):
			blocks = []
			for j in range(self.num_blocks_back[i]):
				if j == 0:
					blocks.append(PostRes(self.nfb[i] * 2, self.nfb[i]))
				else:
					blocks.append(PostRes(self.nfb[i], self.nfb[i]))
			setattr(self, 'backward' + str(4-i), nn.Sequential(*blocks))

		self.drop = nn.Dropout3d(p=0.5, inplace=False)
		self.softmax = nn.Softmax(dim=1)#(NCDHW)

	def forward(self, input,layer1, layer2, layer3, layer4, layer5):
		# decoder
		up4 = self.deconv4(layer5)
		# print('up4.shape',up4.shape)
		# print('layer4.shape',layer4.shape)
		cat_4 = torch.cat((up4, layer4), 1)
		layer_4 = self.backward4(cat_4)
		# layer_4 = self.drop(layer_4)

		up3 = self.deconv3(layer_4)
		cat_3 = torch.cat((up3, layer3), 1)
		layer_3 = self.backward3(cat_3)
		# layer_3 = self.drop(layer_3)

		up2 = self.deconv2(layer_3)
		cat_2 = torch.cat((up2, layer2), 1)
		layer_2 = self.backward2(cat_2)
		# layer_2 = self.drop(layer_2)

		up1 = self.deconv1(layer_2)
		cat_1 = torch.cat((up1, layer1), 1)
		layer_1 = self.backward1(cat_1)
		# layer_1 = self.drop(layer_1)

		layer_1 = self.output(layer_1)
		layer_1 = self.softmax(layer_1)

		return layer_1

class Decoder2(nn.Module):
	def __init__(self):
		super().__init__()
		self.num_blocks_back = [3, 3, 2, 2]  # [5-2]
		self.nff = [1, 8, 16, 32, 64, 128]  # NumFeature_Forw[0-5]
		self.nfb = [64, 32, 16, 8, 2]  # NunFeaturn_Back[5-0]
		#deconv4-1,output
		self.deconv4 = nn.Sequential(
			nn.ConvTranspose3d(self.nff[5], self.nfb[0], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[0]),
			nn.ReLU(inplace=True))
		self.deconv3 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[0], self.nfb[1], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[1]),
			nn.ReLU(inplace=True))
		self.deconv2 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[1], self.nfb[2], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[2]),
			nn.ReLU(inplace=True))
		self.deconv1 = nn.Sequential(
			nn.ConvTranspose3d(self.nfb[2], self.nfb[3], kernel_size=2, stride=2),
			nn.InstanceNorm3d(self.nfb[3]),
			nn.ReLU(inplace=True))
		self.output = nn.Sequential(
			nn.Conv3d(self.nfb[3], self.nfb[3], kernel_size=1),
			nn.InstanceNorm3d(self.nfb[3]),
		    nn.ReLU(inplace=True),
			# nn.Dropout3d(p = 0.3),
		    nn.Conv3d(self.nfb[3], self.nfb[4], kernel_size=1))  # since class number = 3 and split into 2 branch

		#backward4-1
		for i in range(len(self.num_blocks_back)):
			blocks = []
			for j in range(self.num_blocks_back[i]):
				if j == 0:
					blocks.append(PostRes(self.nfb[i] * 2, self.nfb[i]))
				else:
					blocks.append(PostRes(self.nfb[i], self.nfb[i]))
			setattr(self, 'backward' + str(4-i), nn.Sequential(*blocks))

		self.drop = nn.Dropout3d(p=0.5, inplace=False)
		self.softmax = nn.Softmax(dim=1)#(NCDHW)

	def forward(self, input, layer1, layer2, layer3, layer4, layer5):
		# decoder
		up4 = self.deconv4(layer5)
		# print('up4.shape',up4.shape)
		# print('layer4.shape',layer4.shape)
		cat_4 = torch.cat((up4, layer4), 1)
		layer_4 = self.backward4(cat_4)
		# layer_4 = self.drop(layer_4)

		up3 = self.deconv3(layer_4)
		cat_3 = torch.cat((up3, layer3), 1)
		layer_3 = self.backward3(cat_3)
		# layer_3 = self.drop(layer_3)

		up2 = self.deconv2(layer_3)
		cat_2 = torch.cat((up2, layer2), 1)
		layer_2 = self.backward2(cat_2)
		# layer_2 = self.drop(layer_2)

		up1 = self.deconv1(layer_2)
		cat_1 = torch.cat((up1, layer1), 1)
		layer_1 = self.backward1(cat_1)
		# layer_1 = self.drop(layer_1)

		layer_1 = self.output(layer_1)
		layer_1 = self.softmax(layer_1)

		return layer_1

class PiaNet(nn.Module):
	def __init__(self):
		super(PiaNet, self).__init__()
		self. nff = [1, 8, 16, 32, 64, 128]#NumFeature_Forw[0-5]
		self.num_blocks_forw = [2, 2, 3, 3]#[2-5]
		# forward1
		self.forward1 = nn.Sequential(
			nn.Conv3d(self.nff[0], self.nff[1], kernel_size=3, padding=1),
			nn.InstanceNorm3d(self.nff[1]),
			nn.ReLU(inplace=True),
			nn.Conv3d(self.nff[1], self.nff[1], kernel_size=3, padding=1),
			nn.InstanceNorm3d(self.nff[1]),
			nn.ReLU(inplace=True)
		)
		# forward2-5
		for i in range(len(self.num_blocks_forw)):#4
			blocks = []
			for j in range(self.num_blocks_forw[i]):#{2,2,3,3}
				if j == 0:  # conv
					###plus source connection
					blocks.append(PostRes(self.nff[i+1] + 1, self.nff[i+2]))
				else:
					blocks.append(PostRes(self.nff[i+2], self.nff[i+2]))
			setattr(self, 'forward' + str(i + 2), nn.Sequential(*blocks))

		self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
		self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
		self.unmaxpool = nn.MaxUnpool3d(kernel_size=2, stride=2)

		self.decoder1 = Decoder1()
		self.decoder2 = Decoder2()

		self.drop = nn.Dropout3d(p=0.5, inplace=False)

	def forward(self, input):
		#encoder
		# input = input.float()
		layer1 = self.forward1(input)

		down1 = self.maxpool(layer1)
		source1 = self.avgpool(input)
		cat1 = torch.cat((down1, source1), 1)
		layer2 = self.forward2(cat1)

		down2 = self.maxpool(layer2)
		source2 = self.avgpool(source1)
		cat2 = torch.cat((down2, source2), 1)
		layer3 = self.forward3(cat2)
		#layer3 = self.drop(layer3)

		down3 = self.maxpool(layer3)
		source3 = self.avgpool(source2)
		cat3 = torch.cat((down3,source3),1)
		layer4 = self.forward4(cat3)
		# layer4 = self.drop(layer4)

		down4 = self.maxpool(layer4)
		source4 = self.avgpool(source3)
		cat4 = torch.cat((down4,source4),1)
		layer5 = self.forward5(cat4)
		# layer5 = self.drop(layer5)

		branch1 = self.decoder1(input, layer1, layer2, layer3, layer4, layer5)
		branch2 = self.decoder2(input, layer1, layer2, layer3, layer4, layer5)

		return branch1, branch2

def main():
	# net = PiaNet().cuda()#necessary for torchsummary, must to cuda
	# from torchsummary import summary
	# summary(net, input_size=(1,16,256,256))#must remove the number of N

	# input = torch.randn([1,1,16,256,256]).cuda()#(NCDHW)
	# print('############net.named_parameters()#############')
	# for name, param in net.named_parameters():
	# 	print(name)

	gru = SubGRUNet()
	print('############gru.named_parameters()#############')
	for name, param in gru.named_parameters():
		print(name,param.shape)
	# for i in gru.state_dict():
	# 	print(i)

if __name__ == '__main__':
	main()