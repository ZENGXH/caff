import caffe.read_net
import caffe.get_net

import scipy.io as sio
import numpy as np
import re

def main(argv):
	MODEL = argv[1]
	image_name = argv[2]
	#image_name = 'cat'
	# MODEL = 'VGG'
	layer_type = 'all_layer'
	caffe_root = '../'

	layers = caffe.read_net.get(MODEL)
	net_data = caffe.get_net.get(MODEL,image_name)
	
	# print the prediction for classification model
#	print("predicted class is #{}.".format(out['prob'][0].argmax()))	

	netsize = len(layers)
	feat_collect = {}
	# feat_collect = [];

	pattern = re.compile("relu.")	
	pattern_d = re.compile("drop.")
#	layer_relu = 'Rule'
	for layer in layers:
	# do not copy relu layer:
	    if not pattern.match(layer.name) and not pattern_d.match(layer.name):
#	    if layer.type != 'RELU' and layer.type != 'DROPOUT':
        	# feat_collect.append({layer.name,net_data.blobs[layer.name].data[:]})
		feat_collect[layer.name] = net_data.blobs[layer.name].data
	matfile = caffe_root + 'python/_temp/'+MODEL+'_'+layer_type+'_'+image_name+'.mat'

	sio.savemat(matfile,feat_collect)
	print('Done. save as ' + matfile)
if __name__ == '__main__':
	import sys	
	main(sys.argv)
