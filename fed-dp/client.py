
import models, torch, copy
class Client(object):

	def __init__(self, conf, model, train_dataset, id = -1):
		
		self.conf = conf

		self.local_model = models.get_model(self.conf["model_name"], self.conf["n_class"]) 
		
		self.client_id = id
		
		self.train_dataset = train_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
									
		
	def local_train(self, model):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
			
		#print("\n\nlocal model train ... ... ")
		#for name, layer in self.local_model.named_parameters():
		#	print(name, "->", torch.mean(layer.data))
			
		#print("\n\n")
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		
		
		self.local_model.train()
		total_loss = 0.0
		dataset_size = 0
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				#for name, layer in self.local_model.named_parameters():
				#	print(torch.mean(self.local_model.state_dict()[name].data))
				#print("\n\n")
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
				loss.backward()
    
				dataset_size += data.size()[0]
				total_loss += loss.item() # sum up batch loss
			
				optimizer.step()
				
				#for name, layer in self.local_model.named_parameters():
				#	print(torch.mean(self.local_model.state_dict()[name].data))
				#print("\n\n")
				if self.conf["dp"]:
					model_norm = models.model_norm(model, self.local_model)
					
					norm_scale = min(1, self.conf['C'] / (model_norm))
					#print(model_norm, norm_scale)
					for name, layer in self.local_model.named_parameters():
						clipped_difference = norm_scale * (layer.data - model.state_dict()[name])
						layer.data.copy_(model.state_dict()[name] + clipped_difference)
						
			print("Epoch %d done." % e, f"Loss: {total_loss/ dataset_size : .04f}")	
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			
		#print("\n\nfinishing local model training ... ... ")
		#for name, layer in self.local_model.named_parameters():
		#	print(name, "->", torch.mean(layer.data))
		return diff
		