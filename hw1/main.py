# # # # # # # # # # # # # # # # # # 
# UW CSE517 NLP HW1 
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

STOP_SYMBOL = 'STOP'
START_SYMBOL = 'START'
UNK_SYMBOL = 'UNK'
#TRAIN_SET_PATH = '/Users/kevin/Documents/UWEE/NLP/hw1/prob1_brown_full/brown.train.txt'
#TEST_SET_PATH = '/Users/kevin/Documents/UWEE/NLP/hw1/prob1_brown_full/brown.train.txt'
TRAIN_SET_PATH = '/home/titan/CSE517-NLP/HW1/prob1_brown_full/brown.train.txt'
TEST_SET_PATH = '/home/titan/CSE517-NLP/HW1/prob1_brown_full/brown.test.txt'
WORDS_COUNT_THRESHOLD = 1

def remap_keys(mapping):
	return [{'key':k, 'value': v} for k, v in mapping.iteritems()]

def read_dataset(filename):
	with open(filename,'r') as lines:
		dataset = [line.rstrip('\n') for line in lines]
	lines.close()
	return dataset

def non_freq_words_UNK(tokenlist, wordfreq, replaceToken='UNK'):
	return [w if wordfreq[w] > WORDS_COUNT_THRESHOLD else replaceToken for w in tokenlist]


def test_set_UNKing(tokenlist, voc, replaceToken='UNK'):
	return [w if (find_element_in_list(w, voc)==True) else replaceToken for w in tokenlist]

def find_element_in_list(element, list_element):
	try:
		index_element = list_element.index(element)
		return True
	except ValueError:
		return False

def preproc(dataset):
	# add START/STOP symbol
	sentences = []
	for line in dataset:
		#temp = line.split(' ')+[STOP_SYMBOL]
		temp = line.split(' ')+[STOP_SYMBOL]
		sentences.extend(temp)

	# count the number of words
	counts = defaultdict(lambda: 0)
	total_words = 0
	for w in sentences:
		counts[w] = counts[w]+1
		if w != START_SYMBOL:
			total_words = total_words + 1


	vocab_with_oov = [w for w in counts.iteritems()]
	vocab_with_oov.append(UNK_SYMBOL)
	vocab_with_oov_size = len(vocab_with_oov)
	#print 'vocab size with oov'
	#print vocab_with_oov_size

	# replace non-freq words with UNK
	final_corpus = non_freq_words_UNK(sentences, counts, replaceToken='UNK')
	
	#with open('data_with_unk.json', 'w') as f:
    # 		json.dump(final_corpus, f)

	vocab = [w for w,n in counts.iteritems() if n > WORDS_COUNT_THRESHOLD]
	vocab_size = len(vocab)
	vocab.append(UNK_SYMBOL)
	#vocab.remove(START_SYMBOL)
	vocab_size = len(vocab)
	#print total_words
	#print vocab_size
	return (final_corpus,total_words,vocab_size,vocab,vocab_with_oov_size,vocab_with_oov)

def preproc_test(dataset,voc):
	# add START/STOP symbol
	sentences = []
	for line in dataset:
		#temp = line.split(' ')+[STOP_SYMBOL]
		temp = line.split(' ')+[STOP_SYMBOL]
		sentences.extend(temp)

	final_corpus = test_set_UNKing(sentences, voc, replaceToken='UNK')
	return (final_corpus)

def train_NEW(corpus, N):
	#print N # total number of words
	#print V # vocabulary size
	bigram_count = defaultdict(set)  # multiple keys set
	trigram_count = defaultdict(set)
	unigram_count = defaultdict(lambda: 0)
	bigram_model = defaultdict(set)  # multiple keys set
	trigram_model = defaultdict(set)
	unigram_model = defaultdict(lambda: 0)
	beginning_training = time.time();
	idx2 = STOP_SYMBOL
	idx1 = STOP_SYMBOL
	idx0 = STOP_SYMBOL	
	for w in corpus:
		idx0 = w
		if idx1 == STOP_SYMBOL:
			idx2 = START_SYMBOL
			idx1 = START_SYMBOL
			unigram_count[idx1] = unigram_count[idx1] + 1
			bigram_count[idx1, idx2] = bigram_count.get((idx1, idx2), 0) + 1

		unigram_count[idx0] = unigram_count[idx0] + 1
		bigram_count[idx0, idx1] = bigram_count.get((idx0, idx1), 0) + 1
		trigram_count[idx0, idx1, idx2] = trigram_count.get((idx0, idx1, idx2), 0) + 1 
		idx2 = idx1
		idx1 = idx0

	ending_training = time.time();
	print "The training time is : ", ending_training - beginning_training;
	return (unigram_count, bigram_count, trigram_count)

def train_Add_K(corpus, N, V, Voc, K):
	#print N # total number of words
	#print V # vocabulary size
	bigram_count = defaultdict(set)  # multiple keys set
	trigram_count = defaultdict(set)
	bigram_model = defaultdict(set)  # multiple keys set
	trigram_model = defaultdict(set)
	#beginning_training = time.time();
	idx2 = STOP_SYMBOL
	idx1 = STOP_SYMBOL
	idx0 = STOP_SYMBOL	
	for w in corpus:
		idx0 = w
		if idx1 == STOP_SYMBOL:
			idx2 = START_SYMBOL
			idx1 = START_SYMBOL
			bigram_count[idx1, idx2] = bigram_count.get((idx1, idx2), 0) + 1

		bigram_count[idx0, idx1] = bigram_count.get((idx0, idx1), 0) + 1
		trigram_count[idx0, idx1, idx2] = trigram_count.get((idx0, idx1, idx2), 0) + 1 
		idx2 = idx1
		idx1 = idx0

	return (trigram_count,bigram_count)
	

def test(test_set, models, N):
	unigram_count = models[0]
	bigram_count = models[1]
	trigram_count = models[2]
	unigram_prob = defaultdict(lambda: 0)
	bigram_prob = defaultdict(lambda: 0)
	trigram_prob = defaultdict(lambda: 0)
	sentences_count = 0

	idx2 = START_SYMBOL
	idx1 = START_SYMBOL
	idx0 = START_SYMBOL
	uni_prob = 0
	bi_prob = 0
	tri_prob = 0 
	M = 0
	for w in test_set:
		M = M + 1
		idx0 = w
		uni_prob = uni_prob + np.log2(unigram_count[idx0])-np.log2(float(N))
		if idx1 == STOP_SYMBOL:
			idx2 = START_SYMBOL
			idx1 = START_SYMBOL

		bi_prob = bi_prob + np.log2(bigram_count.get((idx0, idx1), 0))-np.log2(float(unigram_count[idx1]))
		tri_prob = tri_prob + np.log2(trigram_count.get((idx0, idx1, idx2), 0))-np.log2(float(bigram_count.get((idx1, idx2), 0)))


		if idx0 == STOP_SYMBOL:
			sentences_count = sentences_count + 1
			unigram_prob[sentences_count] = uni_prob
			bigram_prob[sentences_count] = bi_prob
			trigram_prob[sentences_count] = tri_prob
			uni_prob = 0
			bi_prob = 0
			tri_prob = 0

		idx2 = idx1
		idx1 = idx0

	uni_l = 0
	bi_l = 0
	tri_l = 0
	for s in range(1, sentences_count):	
		uni_l = uni_l + unigram_prob[s]
		bi_l = bi_l + bigram_prob[s]
		tri_l = tri_l + trigram_prob[s]

	uni_perplexity = pow(2,-uni_l/M)
	bi_perplexity = pow(2,-bi_l/M)
	tri_perplexity = pow(2,-tri_l/M)
	print 'uni_perplexity = %f' % uni_perplexity
	print 'bi_perplexity = %f' % bi_perplexity
	print 'tri_perplexity = %f' % tri_perplexity


def interpolation(test_set, models, lambda_set, N):
	unigram_count = models[0]
	bigram_count = models[1]
	trigram_count = models[2]
	trigram_prob = defaultdict(lambda: 0)
	sentences_count = 0


	idx2 = START_SYMBOL
	idx1 = START_SYMBOL
	idx0 = START_SYMBOL

	tri_prob = 0 
	M = 0
	aa = 0
	bb = 0
	cc = 0
	for w in test_set:
		M = M + 1
		idx0 = w
		if idx1 == STOP_SYMBOL:
			idx2 = START_SYMBOL
			idx1 = START_SYMBOL

		aa = unigram_count[idx0]/float(N)
		if bigram_count.get((idx0, idx1), 0)==0:
			bb = 0
		else:
			bb = bigram_count.get((idx0, idx1), 0)/float(unigram_count[idx1])
		if trigram_count.get((idx0, idx1, idx2), 0)==0 or bigram_count.get((idx1, idx2), 0)==0:
			cc = 0
		else:
			cc = trigram_count.get((idx0, idx1, idx2), 0)/float(bigram_count.get((idx1, idx2), 0))
		
		tri_prob = tri_prob + np.log2(aa*lambda_set[0] + bb*lambda_set[1] + cc*lambda_set[2])

		if idx0 == STOP_SYMBOL:
			sentences_count = sentences_count + 1
			trigram_prob[sentences_count] = tri_prob
			tri_prob = 0

		idx2 = idx1
		idx1 = idx0

	tri_l = 0
	for s in range(1, sentences_count):	
		tri_l = tri_l + trigram_prob[s]

	tri_perplexity = pow(2,-tri_l/M)
	print 'interpolation_perplexity (%f,%f,%f) = %f' % (lambda_set[0],lambda_set[1],lambda_set[2],tri_perplexity)

def test_Add_K(test_set, trigram_count, bigram_count, K, V):
	#trigram_model = model
	trigram_prob = defaultdict(lambda: 0)
	sentences_count = 0

	idx2 = START_SYMBOL
	idx1 = START_SYMBOL
	idx0 = START_SYMBOL
	tri_prob = 0 
	M = 0
	for w in test_set:
		M = M + 1
		idx0 = w
		if idx1 == STOP_SYMBOL:
			idx2 = START_SYMBOL
			idx1 = START_SYMBOL

		tri_prob = tri_prob + np.log2(trigram_count.get((idx0, idx1, idx2), 0)+K)-np.log2(float(bigram_count.get((idx1, idx2), 0))+K*V)

		if idx0 == STOP_SYMBOL:
			sentences_count = sentences_count + 1
			trigram_prob[sentences_count] = tri_prob
			tri_prob = 0

		idx2 = idx1
		idx1 = idx0

	tri_l = 0
	for s in range(1, sentences_count):	
		tri_l = tri_l + trigram_prob[s]

	tri_perplexity = pow(2,-tri_l/M)
	print 'K = %f, tri_perplexity = %f' % (K,tri_perplexity)

def prob3_ngram():

	train_corpus = read_dataset(TRAIN_SET_PATH)
	(training_set,N,V,Voc,V_star,Voc_star) = preproc(train_corpus) 	 # N = total words, V = vocab size, Voc = vocabuary
	trained_models = train_NEW(training_set,N)
	#model_validation(trained_models, Voc)
	test_corpus = read_dataset(TEST_SET_PATH)
	test_set = preproc_test(test_corpus,Voc)
	test(test_set, trained_models,N)	

def prob4_add_k():

	train_corpus = read_dataset(TRAIN_SET_PATH)
	test_corpus = read_dataset(TEST_SET_PATH)
	(training_set,N,V,Voc,V_star,Voc_star) = preproc(train_corpus) 	 # N = total words, V = vocab size, Voc = vocabuary
	print "v_size = %d, v_oov_size = %d" % (V, V_star)
	test_set = preproc_test(test_corpus,Voc)
	K_set = [1000000000000, 10, 1, 0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
	for k in K_set:
		(trained_tri,trained_bi) = train_Add_K(training_set,N,V_star,Voc_star,k)
		test_Add_K(test_set, trained_tri, trained_bi, k, V)

def prob4_interpolation():

	train_corpus = read_dataset(TRAIN_SET_PATH)
	test_corpus = read_dataset(TEST_SET_PATH)
	(training_set,N,V,Voc,V_star,Voc_star) = preproc(train_corpus) 	 # N = total words, V = vocab size, Voc = vocabuary
	trained_models = train_NEW(training_set,N)
	test_set = preproc_test(test_corpus,Voc)
	lambda_set = []
	lambda_set.append([0.5,0.3,0.2])
	lambda_set.append([0.8,0.1,0.1])
	lambda_set.append([0.1,0.8,0.1])
	lambda_set.append([0.1,0.1,0.8])
	lambda_set.append([0.6,0.2,0.2])
	lambda_set.append([0.2,0.6,0.2])
	lambda_set.append([0.2,0.2,0.6])
	lambda_set.append([0.4,0.3,0.3])
	lambda_set.append([0.3,0.4,0.3])
	lambda_set.append([0.3,0.3,0.4])
	lambda_set.append([0.2,0.4,0.4])
	lambda_set.append([0.4,0.2,0.4])
	lambda_set.append([0.4,0.4,0.2])
	lambda_set.append([0.1,0.4,0.5])
	lambda_set.append([0.1,0.3,0.6])
	lambda_set.append([0.1,0.2,0.7])
	lambda_set.append([0.05,0.15,0.8])
	lambda_set.append([0.05,0.05,0.9])
	for s in lambda_set:
		interpolation(test_set, trained_models, s, N)

def main():
	# this function train n-gram models and compute the perplexity
	prob3_ngram()	
	# This function will train trigram model with add-k smoothing, and compute the perplexity on the dataset
	#prob4_add_k()
	# This function will train trigram model with linear interpolation, and compute the perplexity on the dataset
	#prob4_interpolation()

if __name__ == "__main__":
	main() 
