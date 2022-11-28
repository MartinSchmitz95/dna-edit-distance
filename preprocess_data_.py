import os
import argparse
import torch
import random

def split(data, test_percentage=0.002):
	centers_path = os.path.join(data, 'Centers.txt')
	clusters_path = os.path.join(data, 'Clusters.txt')

	with open(centers_path) as f:
		centers = f.readlines()
	with open(clusters_path) as f:
		clusters = f.readlines()
	references = []
	for c in centers:
		references.append(c[1:-4])

	test_strands = []
	training_strands = []
	test_references = []
	training_references = []

	test_set = True
	cluster_id = -1

	for l in clusters:
		if l[0] == '=':
			cluster_id += 1
			if random.uniform(0, 1) < test_percentage: # train or test
				test_set = True
				test_references.append(centers[cluster_id])
			else:
				test_set = False
				training_references.append(centers[cluster_id])

		if test_set:
			test_strands.append(l)
		else:
			if l[0] == '=': continue
			training_strands.append(l)

	with open(os.path.join(data, 'test_Centers.txt'), 'w') as f:
		for line in test_references:
			f.write(line)
	with open(os.path.join(data, 'train_Centers.txt'), 'w') as f:
		for line in training_references:
			f.write(line)
	with open(os.path.join(data, 'test_Clusters.txt'), 'w') as f:
		for line in test_strands:
			f.write(line)
	with open(os.path.join(data, 'train_Clusters.txt'), 'w') as f:
		for line in training_strands:
			f.write(line)

def get_data(data):
	centers_path = os.path.join(data, 'train_Centers.txt')
	clusters_path = os.path.join(data, 'train_Clusters.txt')
	# centers_path = os.path.join(data, 'Centers.txt')
	# clusters_path = os.path.join(data, 'Clusters.txt')

	with open(centers_path) as f:
		centers = f.readlines()
	with open(clusters_path) as f:
		clusters = f.readlines()
	references = []
	for c in centers:
		references.append(c[1:-4])

	strands = []
	for l in clusters:
		if l[0] == '=':
			continue
		end = l.find('\\')
		strands.append(l[1:end])
	return references, strands

def save_strands(data, s):
	random.shuffle(s)
	with open(os.path.join('data','shuffled_dna_strings'), 'w') as f:
		for line in s:
			f.write(line)
			f.write('\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# input
	parser.add_argument('--data', type=str, default='clustered-nanopore-reads-dataset/', help='Path to folder with data tuples')

	args = parser.parse_args()
	split(args.data)
	reference, strands = get_data(args.data)
	save_strands(args.data, strands)
