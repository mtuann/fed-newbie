import argparse, json
from email.mime import base
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets
import wandb
from fed_utils import draw_distribution_labels
import numpy as np
import torch
    
def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

sweep_config = {
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'acc_test'},
    'parameters': {
        'model_name': {
            'values': [ 'resnet18', 'resnet34', 'inception_v3']
        },
        'batch_size': {'values': [128]},
        'sample_host': {'values': list(np.linspace(2000, 20000, 10).astype(np.int32)) + list(np.linspace(30000, 50000, 3).astype(np.int32))},
        # [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 50000]
        'no_models': {'values': [10]},
        'type': {'values': ['cifar']},
        'global_epochs': {'values': [200]},
        'local_epochs': {'values': [3]},
        'k': { 'values': [5]},
        'lr': {'values': [0.001]},
        'momentum': {'values': [0.0001]},
        'lambda': {'values': [0.1]},
    }
}


if __name__ == '__main__':
    
	seed_everything(42)
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	
	DIR_MODEL = '/home/vishc2/tuannm/fed/fed-sec/ckpt'
	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	nsam = [2000, 4000, 6000 , 8000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 35000, 40000, 45000, 50000]
	for ns in nsam:
		print(f"Running to nsam: {ns}")
		conf['sample_host'] = ns
  
		train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])	
		server = Server(conf, train_datasets, eval_datasets, conf['sample_host'])
		server.stat_data()
	
		with wandb.init(project='centralized-cifar10-test') as run:
			run.name =f"{conf['model_name']}__{conf['sample_host']}"

			for id_epoch in range(conf['global_epochs']):
				train_loss, train_acc = server.server_epoch_running(id_epoch, f"train-{ns}", optimize=True)
				test_loss, test_acc = server.server_epoch_running(id_epoch, "test", optimize=False)
				if test_acc.avg >= 0.75 or id_epoch == conf['global_epochs'] - 1:
					path_model = f'{DIR_MODEL}/{run.name}__{id_epoch}__{train_acc.avg:.04f}__{test_acc.avg:.04f}.pth'
					server.save_model(path_model=path_model)
					print("Done save model at epoch {} path {}".format(id_epoch, path_model))
				print(f"At epoch {id_epoch} loss_train: {train_loss.avg:.04f} acc_train: {train_acc.avg:.04f} loss_test: {test_loss.avg:.04f} acc_test: {test_acc.avg:.04f}")
				# wandb.log({'epoch': id_epoch, 'loss_train': train_loss.avg, 'acc_train': train_acc.avg,
				# 			'loss_test': test_loss.avg, 'acc_test': test_acc.avg,}, step=id_epoch)
		
	# clients = []
	
	# for c in range(conf["no_models"]):
	# 	clients.append(Client(conf, server.global_model, train_datasets, c))	

	# wandb.init(project='fed-backdoor-attack', name=f"{conf['model_name']}__{conf['type']}__{conf['no_models']}_clients__0_malicious__{conf['global_epochs']}_global_epochs")
	# for e in range(conf["global_epochs"]):
	
	# 	candidates = random.sample(clients, conf["k"])
		
	# 	weight_accumulator = {}
		
	# 	for name, params in server.global_model.state_dict().items():
	# 		weight_accumulator[name] = torch.zeros_like(params)
		
	# 	for c in candidates:
	# 		diff = c.local_train(server.global_model)
			
	# 		for name, params in server.global_model.state_dict().items():
	# 			weight_accumulator[name].add_(diff[name])
				
		
	# 	server.model_aggregate(weight_accumulator)
		
	# 	acc, loss = server.model_eval()
		
	# 	print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
	# 	wandb.log({'epoch': e, 'test_loss': loss, 'test_acc': acc}, step=e)	
			
		
		
	
		
		
	