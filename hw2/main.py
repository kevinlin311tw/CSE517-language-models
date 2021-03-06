# # # # # # # # # # # # # # # # # # 
# UW CSE517 NLP HW2
# Kevin Lin
# # # # # # # # # # # # # # # # # # 

import re
import json
from collections import defaultdict
import os
import time
import math
from math import log
import numpy as np
from sklearn.metrics import confusion_matrix

STOP_SYMBOL = 'STOP'
START_SYMBOL = 'START'
UNK_SYMBOL = 'UNK'
TRAIN_SET_PATH = '/home/titan/CSE517-NLP/hw2/CSE517_HW_HMM_Data/twt.train.json'
DEV_SET_PATH = '/home/titan/CSE517-NLP/hw2/CSE517_HW_HMM_Data/twt.dev.json'
TEST_SET_PATH = '/home/titan/CSE517-NLP/hw2/CSE517_HW_HMM_Data/twt.test.json'
#TRAIN_SET_PATH = '/Users/kevin/Documents/UWEE/NLP/hw2/CSE517_HW_HMM_Data/twt.train.json'
#DEV_SET_PATH = '/Users/kevin/Documents/UWEE/NLP/hw2/CSE517_HW_HMM_Data/twt.dev.json'
#TEST_SET_PATH = '/Users/kevin/Documents/UWEE/NLP/hw2/CSE517_HW_HMM_Data/twt.test.json'
WORDS_COUNT_THRESHOLD = 1
ADD_K = 1

def non_freq_words_UNK(dataset, wordfreq, replaceToken=UNK_SYMBOL):
	for line in dataset:
		for idx in line:
			if wordfreq[idx[0]] <= WORDS_COUNT_THRESHOLD:
				idx[0] = replaceToken
				#idx[1] = replaceToken
	return dataset

def test_set_UNKing(dataset, voc, replaceToken=UNK_SYMBOL):
	for line in dataset:
		for idx in line:
			if (voc[idx[0]] == 0):
				idx[0] = replaceToken
				#idx[1] = replaceToken
	return dataset

def find_element_in_list(element, list_element):
	try:
		index_element = list_element.index(element)
		return True
	except ValueError:
		return False

def read_dataset(filename):
	with open(filename,'r') as lines:
		dataset = [json.loads(line) for line in lines]
	lines.close()
	return dataset

def preproc(dataset):
	# add STOP/START symbol
	for line in dataset:
		#line.insert(0,[START_SYMBOL,START_SYMBOL])
		line.append([STOP_SYMBOL,STOP_SYMBOL])
	# count the number of words
	words_counts = defaultdict(lambda: 0)
	tags_counts =  defaultdict(lambda: 0)
	total_words = 0
	for line in dataset:
		for idx in line:
			w = idx[0] #word
			tag = idx[1] #tag
			words_counts[w] = words_counts[w]+1
			tags_counts[tag] = tags_counts[tag]+1
			total_words = total_words + 1

	vocab_with_oov = [w for w in words_counts.iteritems()]
	vocab_with_oov_size = len(vocab_with_oov)
	print 'vocab size with oov'
	print vocab_with_oov_size
	
	# replace non-freq words with UNK
	final_corpus = non_freq_words_UNK(dataset, words_counts, replaceToken=UNK_SYMBOL)
	
	words_unk_counts = defaultdict(lambda: 0)
	tags_unk_counts =  defaultdict(lambda: 0)
	total_unk_words = 0
	for line in final_corpus:
		for idx in line:
			w = idx[0] #word
			tag = idx[1] #tag
			words_unk_counts[w] = words_unk_counts[w]+1
			tags_unk_counts[tag] = tags_unk_counts[tag]+1
			total_unk_words = total_unk_words + 1

	#vocab_without_oov = [w for w in words_unk_counts.iteritems()]
	#vocab_without_oov_size = len(vocab_without_oov)
	print 'vocab size without oov (with UNK)'
	#print vocab_without_oov_size
	print len(words_unk_counts)
	#print words_unk_counts

	return (final_corpus,words_unk_counts,tags_unk_counts)

def preproc_test(dataset,voc):
	# add STOP/START symbol
	for line in dataset:
		#line.insert(0,[START_SYMBOL,START_SYMBOL])
		line.append([STOP_SYMBOL,STOP_SYMBOL])

	final_corpus = test_set_UNKing(dataset, voc, UNK_SYMBOL)
	return (final_corpus)


def learning(dataset):
	trans_trigram_count = defaultdict(set)
	trans_bigram_count = defaultdict(set)
	trans_unigram_count = defaultdict(lambda: 0)
	emiss_bigram_count = defaultdict(set)
	emiss_unigram_count = defaultdict(lambda: 0)
	beginning_training = time.time();
	# count transitions and emissions
	for line in dataset:
		idx1 = [START_SYMBOL,START_SYMBOL]
		idx2 = [START_SYMBOL,START_SYMBOL]
		for twt in line:
			idx0 = twt	
			if idx0 != START_SYMBOL:
				w0 = idx0[0]
				tag0 = idx0[1]
				w1 = idx1[0]
				tag1 = idx1[1]
				w2 = idx2[0]
				tag2 = idx2[1]
				trans_unigram_count[tag0] = trans_unigram_count[tag0] + 1
				trans_bigram_count[tag0, tag1] = trans_bigram_count.get((tag0, tag1), 0) + 1
				trans_trigram_count[tag0, tag1, tag2] = trans_trigram_count.get((tag0, tag1, tag2), 0) + 1
				emiss_unigram_count[tag0] = emiss_unigram_count[tag0] + 1
				emiss_bigram_count[w0, tag0] = emiss_bigram_count.get((w0, tag0), 0) + 1
			idx2 = idx1
			idx1 = idx0
	
	ending_training = time.time();
	print "The training time is : ", ending_training - beginning_training;	
	return (trans_trigram_count, trans_bigram_count, trans_unigram_count, emiss_bigram_count, emiss_unigram_count)

def display(V):
	# Print a table of steps from dictionary
	yield " ".join(("%12d" % i) for i in range(len(V)))
	for state in V[0]:
		yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


def viterbi(vocab, vocab_tag, words, tags, q_bigram, q_unigram, e_bigram, e_unigram, ADD_K):
	vocab_size = len(vocab)
	V = [{}]

	for t in vocab_tag:
		prob = np.log2(float(e_bigram.get((words[0],t),0)+ADD_K))-np.log2(float(e_unigram[t]+vocab_size*ADD_K))
		V[0][t] = {"prob": prob, "prev": None}
	
	for i in range(1,len(words)):
		V.append({})
		for t in vocab_tag:
			V[i][t] =  {"prob": np.log2(0), "prev": None}
		for t in vocab_tag:
			max_trans_prob = np.log2(0);
			for prev_tag in vocab_tag:
				trans_prob = np.log2(float(q_bigram.get((t, prev_tag),0)+ADD_K))-np.log2(float(q_unigram[prev_tag]+vocab_size*ADD_K))	
				if V[i-1][prev_tag]["prob"]+trans_prob > max_trans_prob:
					max_trans_prob = V[i-1][prev_tag]["prob"]+trans_prob 
					max_prob = max_trans_prob+np.log2(e_bigram.get((words[i],t),0)+ADD_K)-np.log2(float(e_unigram[t]+vocab_size*ADD_K))
					V[i][t] = {"prob": max_prob, "prev": prev_tag}


	opt = []
	previous = None	
	max_prob = max(value["prob"] for value in V[-1].values())

	# Get most probable state and its backtrack
	for st, data in V[-1].items():
		if data["prob"] == max_prob:
			opt.append(st)
			previous = st
			break

	for t in range(len(V) - 2, -1, -1):
		opt.insert(0, V[t + 1][previous]["prev"])
		previous = V[t][previous]["prev"]

	return opt


def compute_accu(predict, ground):
	num = len(ground)
	correct = 0
	total = 0
	for i in range(0,num):
		total = total + 1
		if predict[i]==ground[i]:
			correct = correct + 1

	return (correct,total)



def bigram_inference(vocab, vocab_tag, dataset, q_bigram, q_unigram, e_bigram, e_unigram, ADD_K):
	conf_matrix = defaultdict(set)

	beginning_decoding = time.time();
	total = 0
	hit = 0
	line_counter = 0
	for line in dataset:
		line_counter = line_counter + 1
		words = [w[0] for w in line]
		tags = [t[1] for t in line]
		predict = viterbi(vocab, vocab_tag, words, tags, q_bigram, q_unigram, e_bigram, e_unigram, ADD_K)
		num = len(tags)
		for i in range(0,num):		
			conf_matrix[tags[i], predict[i]] = conf_matrix.get((tags[i], predict[i]),0) + 1

		(correct,count) = compute_accu(predict,tags)
		hit = hit + correct
		total = total + count
		#print 'computing.. (%d/%d): current accuracy = %d/%d = %f' % (line_counter,len(dataset),hit,total,hit/(float)(total))
	print '(add-k k=%d) bigram hmm accuracy = %f (%d/%d)' % (ADD_K,hit/(float)(total),hit,total)
	ending_decoding = time.time();
	#print "The decoding time is : ", ending_decoding - beginning_decoding;	
	'''
	for t in vocab_tag:
		for tt in vocab_tag:
			print '%d ' % conf_matrix.get((t,tt),0)
		print '\n'

	for t in vocab_tag:
		print t
	'''


def prob1_add_k():

	train_corpus = read_dataset(TRAIN_SET_PATH)
	test_corpus = read_dataset(DEV_SET_PATH)
	(train_data, vocab, tags) = preproc(train_corpus)
	test_data = preproc_test(test_corpus,vocab)
	#K_set = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
	K_set = [0]
	(q_trigram, q_bigram, q_unigram, e_bigram, e_unigram) = learning(train_data)
	for ADD_K in K_set:
		bigram_inference(vocab, tags, test_data, q_bigram, q_unigram, e_bigram, e_unigram, ADD_K)


def main():
	prob1_add_k()


if __name__ == "__main__":
	main() 
