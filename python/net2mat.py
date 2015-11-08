import caffe
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

	layers = caffe.Read_net(MODEL)
	net_data = caffe.Get_net(MODEL,image_name)
	para = net_data.params
	parames = {}
	for keys,values in para.items():
	  for i in range(0,len(values)):
	    valSub = values[i]
	    parames['key'+str(i)]=valSub.data
        matfile_para = caffe_root + 'python/_temp/paras'+MODEL+'_'+layer_type+'_'+image_name+'.mat'

        print('parameters collect,',type(parames))
        sio.savemat(matfile_para,parames)
	
	# print the prediction for classification model
#	print("predicted class is #{}.".format(out['prob'][0].argmax()))	

	netsize = len(layers)
	feat_collect = {}
	# feat_collect = [];

	pattern = re.compile("relu.")	
	pattern_d = re.compile("drop.")
#	layer_relu = 'Rule'
	if(MODEL=="AlexNet_SalObjSub" or MODEL== "VGG_ILSVRC_19_layers"):
	  print("load mat from blob dict by key and value")
	  for keys, values in net_data.blobs.items():
	    feat_collect[keys] = values.data	
	else:	
	  for layer in layers:
	# do not copy relu layer:
	      if not pattern.match(layer.name) and not pattern_d.match(layer.name):
#	    if layer.type != 'RELU' and layer.type != 'DROPOUT':
        	# feat_collect.append({layer.name,net_data.blobs[layer.name].data[:]})
	  	if(layer.name == "fc-rcnn"):
			feat_collect["rcnn"] = net_data.blobs[layer.name].data
			continue
		feat_collect[layer.name] = net_data.blobs[layer.name].data

	matfile = caffe_root + 'python/_temp/'+MODEL+'_'+layer_type+'_'+image_name+'.mat'
	
	print('type of collect,',type(feat_collect))
	sio.savemat(matfile,feat_collect)
	print('Done. save as ' + matfile)
if __name__ == '__main__':
	import sys	
	main(sys.argv)
