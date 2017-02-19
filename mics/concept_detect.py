# # # # # # # # # # # # # # # # # #
# UW Kevin Lin
# # # # # # # # # # # # # # # # # #

import json
from collections import defaultdict

if __name__ == '__main__':
	dataset = json.load(open('/home/titan/dataset/coco/dataset.json', 'r'))
	split = defaultdict(list)
	for img in dataset['images']:
		split[img['split']].append(img)
	del dataset
     
	refs = []
    	for img in split['val']:
        	references = [' '.join(tmp['tokens']+['.']) for tmp in img['sentences']]
        	refs.append(references)
    	del split

	print refs[0]

