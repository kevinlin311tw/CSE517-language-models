# # # # # # # # # # # # # # # # # #
# UW Kevin Lin
# # # # # # # # # # # # # # # # # #

import json
from collections import defaultdict
from operator import itemgetter

CONCEPT_NUM = 1000

def generate_file_list(MODE):
	if MODE=='train':
		split_mode = 'train'
	elif MODE=='val':
		split_mode = 'val'
	elif MODE=='test':
		split_mode = 'test'

	dataset = json.load(open('/home/titan/dataset/coco/dataset.json', 'r'))
	split = defaultdict(list)
	for img in dataset['images']:
		split[img['split']].append(img)
	del dataset
     
	refs = []
	refs_img = []
    	for img in split[split_mode]:
        	references = [' '.join(tmp['tokens']) for tmp in img['sentences']]
		filename = img['filename']
        	refs.append(references)
		refs_img.append(filename)

	fname = '%s-file-list.txt' % split_mode
	text_file = open(fname, "w")
	for i in range(0,len(refs_img)):
		filename = refs_img[i]
		if split_mode=='train':
			text_file.write("/home/titan/dataset/coco-new/train2014/%s\n" % filename)
		else:
			text_file.write("/home/titan/dataset/coco-new/val2014/%s\n" % filename)
	text_file.close()


def generate_multilabel_concepts():
	dataset = json.load(open('/home/titan/dataset/coco/dataset.json', 'r'))
	split = defaultdict(list)
	for img in dataset['images']:
		split[img['split']].append(img)
	del dataset
     
	refs = []
	refs_img = []
    	for img in split['train']:
        	references = [' '.join(tmp['tokens']) for tmp in img['sentences']]
		filename = img['filename']
        	refs.append(references)
		refs_img.append(filename)
    	#del split
	
	#print refs[0]
	# count the number of words
	word_freq = defaultdict(lambda: 0)
	for case in refs:
		for line in case:
			for w in line.split(' '):
				#print w
				word_freq[w] += 1

	print len(word_freq)
	ranked_word_freq = sorted(word_freq.iteritems(), key=lambda (k,v): (v,k), reverse=True)

	counter = 0
	concepts_w2idx = defaultdict(lambda: -1)
	concepts_idx2w = {}
	for key, value in ranked_word_freq:
		counter += 1
		if counter<CONCEPT_NUM:
			concepts_w2idx[key] = counter-1
			concepts_idx2w[counter-1] = key
			print "[%d] %s: %s" % (counter,key, value)
		else:
			break

	'''
	for line in refs[0]:
		vec = []
		vec_concept = {}
		filename = refs_img[0]
		print "/home/titan/dataset/coco-new/train2014/%s " % filename
		print line
		for w in line.split(' '):
			idx = concepts_w2idx.get(w,-1)
			if idx>=0:
				vec.append(idx)
				vec_concept[w]=1
		print vec
		print vec_concept
		to_print = [-1]*CONCEPT_NUM
		for item in vec:
			valid = item
			to_print[int(valid)] = 1
		print "%s\n" % ' '.join(map(str, to_print))	
	'''
	
	text_file = open("Output.txt", "w")
	for i in range(0,len(refs)):
		for line in refs[i]:
			vec = []
			filename = refs_img[i]
			text_file.write("/home/titan/dataset/coco-new/train2014/%s " % filename)
			for w in line.split(' '):
				idx = concepts_w2idx.get(w,-1)
				if idx>=0:
					vec.append(idx)
			to_print = [-1]*CONCEPT_NUM
			for item in vec:
				valid = item
				to_print[int(valid)] = 1
			text_file.write("%s\n" % ' '.join(map(str, to_print)))	

	text_file.close()
	

if __name__ == '__main__':
	#generate_file_list('train')
	#generate_file_list('test')
	#generate_file_list('val')
	generate_multilabel_concepts()
	


