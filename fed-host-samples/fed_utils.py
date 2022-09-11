from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

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