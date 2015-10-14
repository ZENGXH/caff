
# coding: utf-8

# # Instant Recognition with Caffe
# 
# In this example we'll classify an image with the bundled CaffeNet model based on the network architecture of Krizhevsky et al. for ImageNet. We'll compare CPU and GPU operation then reach into the model to inspect features and the output.
# 
# (These feature visualizations follow the DeCAF visualizations originally by Yangqing Jia.)

# First, import required modules, 
# set plotting parameters, and 
# run `./scripts/download_model_binary.py 
# models/bvlc_reference_caffenet` 
# to get the pretrained CaffeNet model if it hasn't already been fetched.

import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys

from subprocess import call 
import scipy.io as sio
sys.path.insert(0, caffe_root + 'python')

import caffe
import os
MODELPATH = 'models/bvlc_reference_caffenet/'
MODELFILE = MODELPATH +'bvlc_reference_caffenet.caffemodel'
MODELFILE_DOWNLOAD = '../models/bvlc_reference_caffenet'
deploy_proto = MODELPATH + 'deploy.prototxt'

if not os.path.isfile(caffe_root + MODELFILE):
    print("Downloading pre-trained CaffeNet model...")
    call('../scripts/download_model_binary.py'+ MODELFILE_DOWNLOAD)

# Set Caffe to CPU mode, 
# load the net in the test phase for inference, 
# and configure input preprocessing.

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + deploy_proto,
                caffe_root + MODELFILE,
                caffe.TEST)
mean_file = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + mean_file).mean(1).mean(1)) 
# mean pixel
transformer.set_raw_scale('data', 255)  
# the reference model operates on images in [0,255] range instead of [0,1]

transformer.set_channel_swap('data', (2,1,0))  
# the reference model has channels in BGR order instead of RGB

# Let's start with a simple classification. We'll set a batch of 50 to demonstrate batch processing, even though we'll only be classifying one image. (Note that the batch size can also be changed on-the-fly.)

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)


# Feed in the image (with some preprocessing) and classify with a forward pass.

net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
out = net.forward()
# print("Predicted class is #{}.".format(out['prob'].argmax()))
# print net.blobs['data'].data[0]

# Adorable, but was our classification correct?

# CPU mode
net.forward()  
# call once for allocation

# That's a while, even for a batch size of 50 images. Let's switch to GPU mode.

# First, the layer features and their shapes 
# (1 is the batch size, 
# corresponding to the single input image in this example).

[(k, v.data.shape) for k, v in net.blobs.items()]
"""the output is like
[('data',(20,3,227,227)),
('conv1',(20,96,55,55)),
...
('fc7',(50,4096))]
"""

# The parameters and their shapes. 
# The parameters are 
# `net.params['name'][0]` while biases are `net.params['name'][1]`.
[(k, v[0].data.shape) for k, v in net.params.items()]


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)

    
# The input image
# The first layer filters, `conv1`

# the parameters are a list of [weights, biases]
# filters = net.params['conv1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1))

# The first layer output, `conv1` 
# (rectified responses of the filters above)

feat1 = net.blobs['conv1'].data[0,:]

# The second layer filters, `conv2`# 
# There are 256 filters, each of which has dimension 5 x 5 x 48. We show only the first 48 filters, with each channel shown separately, so that each filter is a row.

# filters = net.params['conv2'][0].data
# vis_square(filters[:48].reshape(48**2, 5, 5))

# The second layer output, 
# `conv2` (rectified, only the first 36 of 256 channels)
feat2 = net.blobs['conv2'].data[0,:]

# The third layer output, `conv3` (rectified, all 384 channels)
feat3 = net.blobs['conv3'].data[0]

# The fourth layer output, `conv4` (rectified, all 384 channels)
feat4 = net.blobs['conv4'].data[0]

# The fifth layer output, `conv5` (rectified, all 256 channels)
feat5 = net.blobs['conv5'].data[0]

# The fifth layer after pooling, `pool5`
feat_p5 = net.blobs['pool5'].data[0]
sio.savemat('../examples/conveMatby00.mat',{'c1':feat1,'c2':feat2,'c2':feat3,'c4':feat4,'c5':feat5,'pool5':feat_p5})


# The first fully connected layer, `fc6` (rectified)# 
#feat = net.blobs['fc6'].data[0]


# The second fully connected layer, `fc7` (rectified)
# feat = net.blobs['fc7'].data[0]


# The final probability output, `prob`
#feat = net.blobs['prob'].data[0]




