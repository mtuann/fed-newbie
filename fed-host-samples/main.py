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
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

def draw_distribution_labels(clients_label, n_label):
	# list of client and labels
	# number of class
	cnt_class = [ [ 0 for j in range(10) ] for i in range(len(clients_label))]
	# client
	txt_client = []
 
	for idc, client in enumerate(clients_label):
		cc = Counter(client)
		for ii in range(n_label):		
			if ii in cc:
				cnt_class[idc][ii] += cc[ii]
		txt_client.append(f'Client {idc:02d}')
  
	print(cnt_class)
	cnt_class = np.array(cnt_class)
	
	fig, ax = plt.subplots()
	fig.set_size_inches(15, 10)
 
	width = 0.35       # the width of the bars: can also be len(x) sequence
	base_cnt = np.array([0 for i in range(len(clients_label))])
	# aa = np.linspace(0.1, 1, 10)
	# base_colors = [f'{a:.1f}' for a in aa ]
	base_colors = ['black', 'red', 'green', 'blue', 'cyan', 'skyblue', 'lightgreen', 'sienna', 'salmon', 'hotpink', 'navy']
	
	for ii in range(n_label):
		ax.bar(txt_client, cnt_class[:, ii], width, color=base_colors[ii], label=f'S{ii}', bottom=base_cnt)
		base_cnt += cnt_class[:, ii]
	fig_title = 'Data Distribution'
	ax.set_title(fig_title)
	ax.legend()
    # fig.savefig('./figures/data_distribution.jpg')
	fig_name = './figures/data_distribution.jpg'
	plt.savefig(fig_name, dpi = 600)
	# pass


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
	
	from collections import Counter
	all_labels = []
 
	for cl in clients:
		lab = []
		for batch_id, batch in enumerate(cl.train_loader):
			_, target = batch
			lab.append(target)
		lab = torch.cat(lab)
  
		all_labels.append(lab.numpy().tolist())
		# print(Counter(lab))
	draw_distribution_labels(all_labels, 10)
	
	# import IPython
	# IPython.embed()
 
	exit(0)
 
	print("\n\n")
	wandb.init(project='fed-backdoor-attack', name=f"{conf['model_name']}__{conf['type']}__{conf['no_models']}_clients__0_malicious__{conf['global_epochs']}_global_epochs")
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
		wandb.log({'epoch': e, 'test_loss': loss, 'test_acc': acc}, step=e)	
			
		
		
	
		
		
	