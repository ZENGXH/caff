#import caffe.read_net
#import caffe.get_net
#import caffe.Segmenter
import caffe
import scipy.io as sio
import numpy as np
import re

import os
import time
import cPickle
import datetime
import logging
# import flask
# import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import caffe.exifutil # from examples/web_demo/exifull.py

# arg: name_of_model name_of_image
def main(argv):
	MODEL_i = argv[1]
	if MODEL_i == 1:
		MODEL = 'bvlc_reference_rcnn_ilsvrc13'
	elif MODEL_i ==2:
		MODEL = 'fcn-8s-pascal'
	elif MODEL_i == 3:
		MODEL = 'fcn-32s-pascal'
	else:
		MODEL = 'fcn-8s-pascal'
	print 'call model '+ MODEL
	image_name = argv[2]

	#image_name = 'cat'
	# MODEL = 'VGG'
	layer_type = 'all_layer'
	caffe_root = '../'
	MODEL_PROTO = caffe_root + 'models/' + MODEL + '/' + MODEL + '.prototxt'
	PRETRAINED = caffe_root + 'models/' + MODEL +'/'+MODEL+'.caffemodel'
	IMAGE_FILE = '../examples/images/'+image_name+'.jpg'
	layers = caffe.Read_net(MODEL)
	# net_data = caffe.get_net.get(MODEL,image_name)
	pallete = [0,0,0,
           128,0,0,
           0,128,0,
           128,128,0,
           0,0,128,
           128,0,128,
           0,128,128,
           128,128,128,
           64,0,0,
           192,0,0,
           64,128,0,
           192,128,0,
           64,0,128,
           192,0,128,
           64,128,128,
           192,128,128,
           0,64,0,
           128,64,0,
           0,192,0,
           128,192,0,
           0,64,128,
           128,64,128,
           0,192,128,
           128,192,128,
           64,64,0,
           192,64,0,
           64,192,0,
           192,192,0]
           
	net_data = caffe.Get_net(MODEL,image_name)
	
	# print the prediction for classification model
#	print("predicted class is #{}.".format(out['prob'][0].argmax()))	
	input_image = 255 * caffe.exifutil.open_oriented_im(IMAGE_FILE)   

	# save 
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

#########################################
	net = caffe.Segmenter(MODEL_PROTO, PRETRAINED, gpu=False)
	# Mean values in BGR format
	mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
	reshaped_mean_vec = mean_vec.reshape(1,1,3);

	# Rearrange channels to form BGR
	im = input_image[:,:,::-1]

	# Subtract mean
	im = im - reshaped_mean_vec

	# Pad as necessary
	cur_h, cur_w, cur_c = im.shape
	pad_h = 500 - cur_h
	pad_w = 500 - cur_w
	im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

	# Get predictions
	segmentation = net.predict([im])

	output_im = Image.fromarray(segmentation)

	output_im.putpalette(pallete);
	imgfile = caffe_root + 'python/_temp/'+MODEL+'_'+layer_type+'_'+image_name+'.png'
	output_im.save(imgfile)


if __name__ == '__main__':
	import sys	
	main(sys.argv)
