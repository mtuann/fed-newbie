
import models, torch


class Server(object):
	
	def __init__(self, conf, eval_dataset):
	
		self.conf = conf 
		
		self.global_model = models.get_model(self.conf["model_name"], self.conf["n_class"]) 
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
			if self.conf['dp']:
				sigma = self.conf['sigma']
				if torch.cuda.is_available():
					noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
				else:
					noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)
					
				update_per_layer.add_(noise)
			
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
				
	def model_eval(self):
		self.global_model.eval()
		#print("\n\nstart to model evaluation......")
		#for name, layer in self.global_model.named_parameters():
		#	print(name, "->", torch.mean(layer.data))
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]
			
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
				
			
			output = self.global_model(data)
			
			#print(output)
			# import IPython
			# IPython.embed()
   
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			# print(batch_id, total_loss)
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		return acc, total_l