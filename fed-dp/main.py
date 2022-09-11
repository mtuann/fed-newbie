import argparse, json
from calendar import EPOCH
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets
import wandb
	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	# print("\n\n")
 	
	with wandb.init(project='fl-dp-cifar10') as run:
		run.name =f"{conf['model_name']}__{conf['no_models']}_clients"

		for e in range(conf["global_epochs"]):

			candidates = random.sample(clients, conf["k"])
			
			weight_accumulator = {}
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name] = torch.zeros_like(params)
			
			for c in candidates:
				diff = c.local_train(server.global_model)
				
				for name, params in server.global_model.state_dict().items():
					weight_accumulator[name].add_(diff[name])
					
			
			server.model_aggregate(weight_accumulator)
			
			acc, loss = server.model_eval()
			
			print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
			wandb.log({'epoch': e, 'loss': loss, 'acc': acc}, step=e)
				
			
		
		
	
		
		
	