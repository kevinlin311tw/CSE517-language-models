# # # # # # # # # # # # # # # # # #
# UW Kevin Lin
# # # # # # # # # # # # # # # # # #

import re
import json
from collections import defaultdict
import os
import time
import math
from math import log
import numpy as np
from glob import glob
import multiprocessing

CONFIG_PATH = '/home/titan/dataset/dataset/preproc/accv_split_details.json'
DATA_PATH = '/kevindisk/data/new-*.json'
IMG_PATH = '/kevindisk/data/img/*.jpg'
listings = []
train = []
test = []
val = []

table_id2idx = defaultdict(lambda: 0)

def find(dataset, spec):
	found = False
	for subset in dataset:
		for item in subset:
			#print item['listing_id'],spec
			if spec == item['listing_id']:
				print 'matching %d'%item['listing_id']
				found = item
	return found
    #return [item for item in dataset if spec in item]

def construct_table(dataset,table):
	counter = 0
	for item in dataset:
		id_num = item['listing_id']
		table_id2idx[id_num] = counter
		counter += 1
	return table
	
def selecting_data(filename, selected_subset, database, table):
	new_subset = []
	beginning_decoding = time.time();
	counter = 0
	total_count = len(selected_subset)
	for idx in selected_subset:
		counter += 1
		print 'process: %d/%d'%(counter,total_count)
		list_idx = table[int(idx)]
		item = database[list_idx]
		if item!=None:
			new_subset.append(item)
	ending_decoding = time.time();
	print "The decoding time is : ", ending_decoding - beginning_decoding;
	# Writing JSON data
	with open(filename, 'w') as f:
		json.dump(new_subset, f)
	print 'write json file: %s'%filename		

def read_dataset(config, data_path):
	f = open(config,'r')
	data = json.load(f)
	global train
	train = data['train']
	global val
	val = data['val']
	global test
	test = data['test']

	for json_file in glob(data_path):
    		with open(json_file, 'r') as f:
			print 'reading..%s'%json_file
			global listings
        		new_list = json.load(f)
			listings = listings + new_list
	
	global table_id2idx
	table_id2idx = construct_table(listings,table_id2idx)

	selecting_data('new_train.json', train, listings, table_id2idx)
	selecting_data('new_val.json', val, listings, table_id2idx)
	selecting_data('new_test.json', test, listings, table_id2idx)


def main():
	read_dataset(CONFIG_PATH, DATA_PATH)

if __name__ == "__main__":
	main()
