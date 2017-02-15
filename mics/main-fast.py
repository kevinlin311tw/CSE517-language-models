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
import cPickle
import random

CONFIG_PATH = '/home/titan/dataset/etsy-dataset/preproc/accv_split_details.json'
DATA_PATH = '/kevindisk/etsy_data/new-*.json'
IMG_PATH = '/kevindisk/etsy_data/img/*.jpg'
GLOVE_PATH = '/home/titan/dataset/glove.6B.300d.txt'
listings = []
train = []
test = []
val = []

table_id2idx = defaultdict(lambda: 0)


def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

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

def data_formulation(input_data, word2idx):
	caption_vectors = []
	filename_vectors = []
	datanum_vectors = []
	for idx, item in enumerate(input_data, start=0):
		vector = []
		desc = item['desc']
		datanum_vectors.append(idx)
		filename_vectors.append(item['listing_id'])
		words = desc.split()
		for w in words:
			vocab = w.strip(string.punctuation)
			num = word2idx[vocab]
			vector.append(num)
		caption_vectors.append(vector)
	dataset = [caption_vectors, datanum_vectors, filename_vectors]
	return (dataset)

def generate_vocab():
	table_word2idx = defaultdict(lambda: 0)
	table_idx2word = {}
	#filename = ['temp.json']#['new_train.json','new_val.json','new_test.json']
	filename = ['new_train.json','new_val.json','new_test.json']
	f1 = open(filename[0],'r')
	f2 = open(filename[1],'r')
	f3 = open(filename[2],'r')
	train = json.load(f1)
	val = json.load(f2)
	test = json.load(f3)
	counter = 0
	for item in train:
		desc = item['desc']
		words = desc.split() 
		#print words
		for w in words:
			vocab = w.strip(string.punctuation)
			if(table_word2idx.get(vocab,0)==0):
				table_word2idx[vocab] = counter
				table_idx2word[int(counter)] = vocab
				counter += 1

	#print table_idx2word[0]
	#table_idx2word = {int(v): k for k, v in table_word2idx.iteritems()}
	with open('table_word2idx.json', 'wb') as f:
		json.dump(table_word2idx, f)
	with open('table_idx2word.json', 'wb') as f:
		json.dump(table_idx2word, f)

	train_set = data_formulation(train, table_word2idx)
	val_set = data_formulation(val, table_word2idx)
	test_set = data_formulation(test, table_word2idx)

	final_data = [train_set]
	final_data.append(val_set)
	final_data.append(test_set)
	final_data.append(table_word2idx)
	final_data.append(table_idx2word)
	
	with open('data.p', 'w') as handle:
    		json.dump(final_data, handle)


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

def read_word2vec(filename):
	counter = 0
	vocab = defaultdict(lambda: 0)
	with open(filename,'r') as lines:
		w2v = [line.rstrip('\n') for line in lines]
	lines.close()
	word_feat = []
	for line in w2v:
		temp = line.split(' ')
		feat = []
		#vocab[counter] = temp[0]
		vocab[temp[0]] = counter
		for w in temp[1:]:
			feat.append(float(w))
		word_feat.append(feat)
		counter += 1
	return (vocab,word_feat)
'''
def proc_word2vec():
	(glove_voc,glove_w2v) = read_word2vec(GLOVE_PATH)
	selected_w2v = []
	selected_voc = defaultdict(lambda: 0)	 
	f = open('table_word2idx.json', 'rb')
	word2idx = json.load(f)
	f2 = open('table_idx2word.json', 'rb')
	idx2word = json.load(f2, object_hook=jsonKeys2int)
	print 'vocb size = %d' % len(idx2word)
	for i in range(1,len(idx2word)):
		print i
		w = idx2word[i]
		selected_voc[w] = i
		glove_indx = glove_voc.get(w,False)
		if glove_indx == False:
			glove_feat = [random.uniform(-0.5,0.5) for _ in range (300)]
		else:
			glove_feat = glove_w2v[glove_indx]
		selected_w2v.append(glove_feat)

	final_w2v = [selected_w2v] 
	final_w2v.append(selected_voc)

	with open('word2vec.p', 'wb') as handle:
    		json.dump(final_w2v, handle)
'''
def proc_word2vec():
	(glove_voc,glove_w2v) = read_word2vec(GLOVE_PATH)
	selected_w2v = []
	selected_voc = {}	 
	f = open('table_word2idx.json', 'rb')
	word2idx = json.load(f)
	f2 = open('table_idx2word.json', 'rb')
	idx2word = json.load(f2, object_hook=jsonKeys2int)
	print 'vocb size = %d' % len(idx2word)
	for i in range(1,len(idx2word)):
		print i
		w = idx2word[i]
		selected_voc[w] = i
		print selected_voc[w]
		glove_indx = glove_voc.get(w,False)
		if glove_indx == False:
			glove_feat = np.asarray([random.uniform(-0.5,0.5) for _ in range (300)])
		else:
			glove_feat = np.asarray(glove_w2v[glove_indx])
		selected_w2v.append(glove_feat)


	saved_data = (selected_w2v,selected_voc)

	with open('word2vec.p', 'wb') as outfile:
    		cPickle.dump(saved_data, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
	'''
	final_w2v = [selected_w2v] 
	final_w2v.append(selected_voc)
	'''
	#with open('word2vec.p', 'wb') as handle:
    	#	json.dump(saved_data, handle)	
	
	
def main():
	#extract_data_from_dataset(CONFIG_PATH, DATA_PATH)
	#generate_img_list()
	#generate_vocab()
	proc_word2vec()
	
if __name__ == "__main__":
	main()
