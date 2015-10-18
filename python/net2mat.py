import caffe.read_net
import caffe.get_net

import scipy.io as sio
import numpy as np

def main():
	image_name = 'cat'
	MODEL = 'bvlc_reference_caffenet'
	layer_type = 'Convolution'
	caffe_root = '../'

	layers = caffe.read_net.get(MODEL)
	net_data = caffe.get_net.get(MODEL,image_name)

	netsize = len(layers)
	feat_collect = {}
	# feat_collect = [];
	for layer in layers:
	    if layer.type == layer_type:
        	# feat_collect.append({layer.name,net_data.blobs[layer.name].data[:]})
		feat_collect[layer.name] = net_data.blobs[layer.name].data
	matfile = caffe_root + 'python/_temp/'+MODEL+'_'+layer_type+'_'+image_name+'.mat'

	sio.savemat(matfile,feat_collect)
	print('Done. save as ' + matfile)
if __name__ == '__main__':
	main()
