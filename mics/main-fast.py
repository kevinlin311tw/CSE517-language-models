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
import string

CONFIG_PATH = '/home/titan/dataset/etsy-dataset/preproc/accv_split_details.json'
DATA_PATH = '/kevindisk/etsy_data/new-*.json'
IMG_PATH = '/kevindisk/etsy_data/img/*.jpg'
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

def extract_data_from_dataset(config, data_path):
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

def generate_vocab():
	table_word2idx = defaultdict(lambda: 0)
	table_idx2word = defaultdict(lambda: 0)	
	filename = ['temp.json']#['new_train.json','new_val.json','new_test.json']
	f1 = open(filename[0],'r')
	#f2 = open(filename[1],'r')
	#f3 = open(filename[2],'r')
	train = json.load(f1)
	#val = json.load(f2)
	#test = json.load(f3)
	counter = 0
	for item in train:
		desc = item['desc']
		words = desc.split() 
		print words
		for w in words:
			vocab = w.strip(string.punctuation)
			if(table_word2idx.get(vocab,0)==0):
				table_word2idx[vocab] = counter
				#table_idx2word[counter] = vocab
				counter += 1

	table_idx2word = {v: k for k, v in table_word2idx.iteritems()}
	with open('table_word2idx.json', 'w') as f:
		json.dump(table_word2idx, f)
	with open('table_idx2word.json', 'w') as f:
		json.dump(table_idx2word, f)

	caption_vectors = []
	filename_vectors = []
	datanum_vectors = []
	for item in train:
		vector = []
		desc = item['desc']
		words.plit()
		for w in words:
			vocab = w.strip(string.punctuation)
			num = table_word2idx[vocab]
			vector.append(num)
		caption_vectors.append(vector)
			
			
		



def generate_img_list():
	filename = ['new_train.json','new_val.json','new_test.json']
	txtname = ['train-file-list.txt','val-file-list.txt','test-file-list.txt']
	for fn in range(0,len(filename)):
		text_file = open(txtname[fn], "w")
		f = open(filename[fn],'r')
		dataset = json.load(f)
		for item in dataset:
			text_file.write("%s\n" % item['filepath'])
		text_file.close()
		f.close()
	
def main():
	#extract_data_from_dataset(CONFIG_PATH, DATA_PATH)
	#generate_img_list()
	generate_vocab()
if __name__ == "__main__":
	main()
