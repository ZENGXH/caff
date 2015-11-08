# coding: utf-8

# # Instant Return of features of differnet layer with Caffe
# 
# input an image with the bundled CaffeNet model based on the pretrained model 
# collect the output 'features' of defferent layer and save as .mat file
# First, import required modules, 
# set plotting parameters, and 
# run `./scripts/download_model_binary.py 
# models/bvlc_reference_caffenet` 
# to get the pretrained CaffeNet model if it hasn't already been fetched.

import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in caffe/python/caffe
import sys

from subprocess import call 
import scipy.io as sio
sys.path.insert(0, caffe_root + 'python')
import json
import caffe
import os
# image_name = 'cat'
def Get_net(MODEL,image_name):
	image_input = caffe_root + 'examples/images/' + image_name + '.jpg'
#	MODEL = 'bvlc_reference_caffenet'

        #'../../models/bvlc_reference_caffenet'
	MODELFILE_DOWNLOAD = caffe_root+ 'models/'+ MODEL
	MODELPATH = MODELFILE_DOWNLOAD + '/'
	MODELFILE = MODELPATH + MODEL +'.caffemodel'
	deploy_proto = MODELPATH + 'deploy.prototxt'
	print("finding: ",MODELFILE)
	if not os.path.isfile(MODELFILE):
	    print("Call and Downloading pre-trained CaffeNet model",MODELFILE_DOWNLOAD)
	    call([caffe_root + 'scripts/download_model_binary.py',MODELFILE_DOWNLOAD])

	# Set Caffe to CPU mode, 
	# load the net in the test phase for inference, 
	# and configure input preprocessing.

	caffe.set_mode_cpu()
	net = caffe.Net(deploy_proto,
	                MODELFILE,
	                caffe.TEST)
	para = net.params
	for keys,values in para.items():
	  print(keys)
	 # print(values.data)
	#print para.key[::2]
	#print para
	#pprint(para,indent4)
	#print(json.dumps(data,indent=4))
	mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

	# input preprocessing: 'data' is the name of the input blob == net.inputs[0]

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	print('data layer shape:',net.blobs['data'].data.shape)

	transformer.set_transpose('data', (2,0,1))

	transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) 

	# mean pixel

	# the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_raw_scale('data', 255)  

	# the reference model has channels in BGR order instead of RGB
	transformer.set_channel_swap('data', (2,1,0))  

	# Let's start with a simple classification. We'll set a batch of 50 to demonstrate batch processing, even though we'll only be classifying one image. (Note that the batch size can also be changed on-the-fly.)

	# set net to batch size of 50
	# net.blobs['data'].reshape(50,3,227,227)

	im = caffe.io.load_image(image_input)
	if(MODEL=="VGG_ILSVRC_19_layers"):
	  net.blobs['data'].reshape(10,3,224,224)
	  im = caffe.io.resize_image(im, (224,224))
	else:
	  net.blobs['data'].reshape(50,3,227,227)
	# Feed in the image (with some preprocessing) and classify with a forward pass.
	net.blobs['data'].data[...] = transformer.preprocess('data', im)
	out = net.forward()
	# print("Predicted class is #{}.".format(out['prob'].argmax()))
	# print net.blobs['data'].data[0]

	# CPU mode
	net.forward()  
	# call once for allocation
	return net




