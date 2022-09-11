
from multiprocessing import reduction
import models, torch
from tqdm import tqdm
from fed_utils import AverageMeter

class Server(object):
	
	def __init__(self, conf, train_dataset, eval_dataset, n_dtrain=2000):
	
		self.conf = conf 
		
		self.global_model = models.get_model(self.conf["model_name"]) 
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
		self.train_dataset = train_dataset
		all_range = list(range(len(self.train_dataset)))
		# data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[: n_dtrain] # all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
		# self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.conf['lr'],
		# 							momentum=self.conf['momentum'])
		self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.conf['lr'])
		
	def save_model(self, path_model):
		torch.save(self.global_model.state_dict(), path_model)
  		# pass
 
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)

	def stat_data(self):
		from collections import Counter
		lab = []
		for batch_id, batch in enumerate(self.train_loader):
			_, target = batch
			lab.append(target)
		lab = torch.cat(lab)
		print("train_loader", Counter(lab.numpy().tolist()))
  
		lab = []
		for batch_id, batch in enumerate(self.eval_loader):
			_, target = batch
			lab.append(target)
		lab = torch.cat(lab)
		print("eval_loader", Counter(lab.numpy().tolist()))
  
	def server_epoch_running(self, epoch, msg="train/test", optimize=True):
		
		self.global_model.train() if optimize else self.global_model.eval()
		data_loader = self.train_loader if optimize else self.eval_loader
		loss_up, acc_up = AverageMeter(), AverageMeter()
		dic_metrics= {'loss':0, 'acc':0, 'lr':0}
		
		with tqdm(total=len(data_loader),desc="Epoch {} - {}".format(epoch, msg)) as pbar:
			for batch_id, batch in enumerate(data_loader):
				data, target = batch

				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				if optimize:
					self.optimizer.zero_grad()
		
				output = self.global_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				if optimize:
					loss.backward()
					dic_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
					self.optimizer.step()
				pred = output.data.max(1)[1]  # get the index of the max log-probability
				correct = pred.eq(target.data.view_as(pred)).cpu().sum().item() / data.shape[0]
				loss_up.update(loss.item(), data.shape[0])
				acc_up.update(correct, data.shape[0])
				dic_metrics['loss'] = loss_up.avg
				dic_metrics['acc'] = acc_up.avg
				pbar.update(1)
				pbar.set_postfix(dic_metrics)

		return loss_up, acc_up
		
  	
	# def model_eval(self):
	# 	self.global_model.eval()
		
	# 	total_loss = 0.0
	# 	correct = 0
	# 	dataset_size = 0
	# 	for batch_id, batch in enumerate(self.eval_loader):
	# 		data, target = batch 
	# 		dataset_size += data.size()[0]
			
	# 		if torch.cuda.is_available():
	# 			data = data.cuda()
	# 			target = target.cuda()
				
			
	# 		output = self.global_model(data)
			
	# 		total_loss += torch.nn.functional.cross_entropy(output, target,
	# 										  reduction='sum').item() # sum up batch loss
	# 		pred = output.data.max(1)[1]  # get the index of the max log-probability
	# 		correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

	# 	acc = 100.0 * (float(correct) / float(dataset_size))
	# 	total_l = total_loss / dataset_size

	# 	return acc, total_l