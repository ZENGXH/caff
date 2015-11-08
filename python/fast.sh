#!/usr/bin/env sh
# This script converts the net data to mat file

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb
#MODEL=bvlc_reference_rcnn_ilsvrc13
MODEL=VGG_ILSVRC_19_layers

#MODEL=AlexNet_SalObjSub
IMAGE=cat

echo "RUNNING $MODEL..."

python net2mat.py  $MODEL $IMAGE




echo "Done."
