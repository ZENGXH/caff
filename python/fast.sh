#!/usr/bin/env sh
# This script converts the net data to mat file

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb
MODEL1=bvlc_reference_rcnn_ilsvrc13
MODEL2=VGG_ILSVRC_19_layers

MODEL3=AlexNet_SalObjSub
IMAGE=cow

echo "RUNNING $MODEL..."

python net2mat.py  $MODEL1 $IMAGE
python net2mat.py  $MODEL2 $IMAGE
python net2mat.py  $MODEL3 $IMAGE




echo "Done."
