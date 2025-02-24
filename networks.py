import torch
import torch.nn as nn
import torch.nn.functional as F

POOL = nn.AvgPool1d


class TwoLayerPool(nn.Module):
	def __init__(self, C, M, embedding, mtc_input):
		super(TwoLayerPool, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.pool = nn.Sequential(nn.MaxPool1d(16), nn.MaxPool1d(16), )
		self.flat_size = C * M // 256

	def forward(self, x: torch.Tensor):
		if len(x.shape) == 3:
			N, C, M = x.shape
		else:
			N = 1
			C, M = x.shape

		x = x.view(N, C, M)
		x = self.pool(x)
		x = x.view(N, self.flat_size)

		return x

class MyCNN_(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(MyCNN, self).__init__()

		kernel = 3
		base_channels = 32
		pooling_size = 2
		drop_out = 0.1

		size = 128

		# 1 x 128
		self.conv1 = nn.Sequential(
			nn.Conv1d(4, base_channels, kernel_size=kernel),
			nn.BatchNorm1d(base_channels),
			nn.ReLU(),
			nn.MaxPool1d(pooling_size, stride=pooling_size))

		# base_channel x 64
		self.conv2 = nn.Sequential(
			nn.Conv1d(base_channels, base_channels, kernel_size=kernel),
			nn.BatchNorm1d(base_channels),
			nn.ReLU(),
			nn.MaxPool1d(pooling_size, stride=pooling_size))

		# base_channel x 32
		self.conv3 = nn.Sequential(
			nn.Conv1d(base_channels, 2 *base_channels, kernel_size=kernel),
			nn.BatchNorm1d(2 * base_channels),
			nn.ReLU(),
			nn.MaxPool1d(pooling_size, stride=pooling_size))

		# base_channel x 16
		self.conv4 = nn.Sequential(
			nn.Conv1d(2 *base_channels, 2 * base_channels, kernel_size=kernel),
			nn.BatchNorm1d(2 * base_channels),
			nn.ReLU(),
			nn.MaxPool1d(pooling_size, stride=pooling_size))

		# base_channel x 8
		self.conv5 = nn.Sequential(
			nn.Conv1d(2 * base_channels, 4 * base_channels, kernel_size=kernel),
			nn.BatchNorm1d(4 * base_channels),
			nn.ReLU(),
			nn.MaxPool1d(pooling_size, stride=2),
			nn.Dropout(drop_out))

		# 4 x 4 x base_channel4 * base_channels * 2
		self.fc1 = nn.Linear(4 * base_channels * 2, 4 * base_channels * 2)
		self.fc2 = nn.Linear(4 * base_channels * 2, embedding)
		self.activation = nn.Sigmoid()
		#  RuntimeError: mat1 and mat2 shapes cannot be multiplied (655350x256 and 512x8)

	def forward(self, x):
		#print(x.shape)
		#x = x.view(x.shape[0], 1, -1)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = out.view(x.shape[0], out.size(1) * out.size(2))
		out = self.fc1(out)
		out = torch.relu(out)
		logit = self.fc2(out)
		logit = self.activation(logit)

		return logit

class MyCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(MyCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = 4
		channel = 128

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			#nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			#POOL(2),
		)

		# Size after pooling
		print(M, C, self.mtc_input, channel)  # 114 4 1 8  # 30 26 1 8
		self.flat_size = channel #M // 1024 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc2 = nn.Linear(self.flat_size, self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x: torch.Tensor):
		#N = len(x)
		#x = x.view(-1, 4, self.M)
		x = self.conv(x)
		# print(x.size())
		x = x.view(x.shape[0], x.shape[1] * x.shape[2])
		x = self.fc2(x)
		x = torch.relu(x)
		x = self.fc1(x)

		return x

class MultiLayerCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(MultiLayerCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
		)

		# Size after pooling
		print(M, C, self.mtc_input, channel)  # 114 4 1 8  # 30 26 1 8
		self.flat_size = M // 1024 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc2 = nn.Linear(self.flat_size, self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x: torch.Tensor):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)
		print(x.shape)
		x = self.conv(x)
		# print(x.size())
		x = x.view(N, self.flat_size)
		x = self.fc2(x)
		x = torch.relu(x)
		x = self.fc1(x)

		return x


class RandomCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(RandomCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
		)

		# Size after pooling
		self.flat_size = M // 1024 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x: torch.Tensor):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)

		x = self.conv(x)
		# print(x.size())
		x = x.view(N, self.flat_size)
		# x = self.fc1(x)
		return x


class EnronCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(EnronCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
		)

		# Size after pooling
		self.flat_size = M // 1024 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x: torch.Tensor):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)

		x = self.conv(x)
		x = x.view(N, self.flat_size)
		x = self.fc1(x)

		return x


class TrecCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(TrecCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
		)

		# Size after pooling
		self.flat_size = M // 1024 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x: torch.Tensor):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)

		x = self.conv(x)
		x = x.view(N, self.flat_size)
		x = self.fc1(x)

		return x


class UnirefCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(UnirefCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.ReLU(),

			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.ReLU(),

			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.ReLU(),

			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
		)

		# Size after pooling
		self.flat_size = M // 4096 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)
		# self.fc2 = nn.Linear(self.flat_size, self.flat_size)

	def forward(self, x: torch.Tensor):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)

		x = self.conv(x)
		x = x.view(N, self.flat_size)
		# x = self.fc2(x)
		# x = torch.relu(x)
		x = self.fc1(x)

		return x


class DBLPCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(DBLPCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
		)

		# Size after pooling
		self.flat_size = M // 1024 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x: torch.Tensor):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)

		x = self.conv(x)
		x = x.view(N, self.flat_size)
		x = self.fc1(x)

		return x


class QuerylogCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(QuerylogCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv = nn.Sequential(
			nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
			nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
			POOL(2),
		)

		# Size after pooling
		self.flat_size = M // 128 * C // self.mtc_input * channel
		print("# self.flat_size ", self.flat_size)
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x: torch.Tensor):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)

		x = self.conv(x)
		# print(x.size())
		x = x.view(N, self.flat_size)
		x = self.fc1(x)

		return x


class TwoLayerCNN(nn.Module):
	def __init__(self, C, M, embedding, channel, mtc_input):
		super(TwoLayerCNN, self).__init__()
		self.C = C
		self.M = M
		self.embedding = embedding
		self.mtc_input = C if mtc_input else 1

		self.conv1 = nn.Conv1d(self.mtc_input, channel, 3, 1, padding=1, bias=False)
		self.flat_size = M // 2 * C // self.mtc_input * channel
		self.fc1 = nn.Linear(self.flat_size, embedding)

	def forward(self, x):
		N = len(x)
		x = x.view(-1, self.mtc_input, self.M)

		x = F.relu(self.conv1(x))
		x = F.max_pool1d(x, 2)

		x = x.view(N, self.flat_size)
		x = self.fc1(x)

		return x


class TripletNet(nn.Module):
	def __init__(self, embedding_net):
		super(TripletNet, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, x):
		x1, x2, x3 = x
		return self.embedding_net(x1), self.embedding_net(x2), self.embedding_net(x3)

	def compute_embs(self, x):
		return self.embedding_net(x)

class TripletLoss(nn.Module):
	def __init__(self, args):
		super(TripletLoss, self).__init__()
		self.l, self.r = 1, 1
		step = args.epochs // 5
		self.Ls = {
			step * 0: (0, 10),
			step * 1: (10, 10),
			step * 2: (10, 1),
			step * 3: (5, 0.1),
			step * 4: (1, 0.01),
		}

	def dist(self, ins, pos):
		return torch.norm(ins - pos, dim=1)

	def forward(self, x, lens, dists, epoch):
		if epoch in self.Ls:
			self.l, self.r = self.Ls[epoch]
		anchor, positive, negative = x
		pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

		pos_embed_dist = self.dist(anchor, positive)
		neg_embed_dist = self.dist(anchor, negative)
		pos_neg_embed_dist = self.dist(positive, negative)

		threshold = neg_dist - pos_dist
		rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

		mse_loss = (pos_embed_dist - pos_dist) ** 2 + \
		           (neg_embed_dist - neg_dist) ** 2 + \
		           (pos_neg_embed_dist - pos_neg_dist) ** 2

		return torch.mean(rank_loss), \
		       torch.mean(mse_loss), \
		       torch.mean(self.l * rank_loss +
		                  self.r * torch.sqrt(mse_loss))
