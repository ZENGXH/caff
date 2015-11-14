#!/usr/bin/env python
"""help function:
usage: import caffe.read_net
layers = caffe.read_net('bvlc_reference_caffenet')

read the deploy.prototxt file and generate the a list of layers
by layers[i].name/type, we can easily get the imformation of the net
"""
from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2

PATH2MODEL = '../models/'
prototxt = '/deploy.prototxt'


def Read_net(MODEL):
    input_net_proto_file = PATH2MODEL + MODEL + prototxt

    caffe_net = caffe_pb2.NetParameter()

    text_format.Merge(open(input_net_proto_file).read(), caffe_net)

    print('reading net from %s' % input_net_proto_file)

    layers = [];

    for layer in caffe_net.layer:
	layers.append(layer)	
    #return layers
    return caffe_net
