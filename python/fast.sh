#!/usr/bin/env sh
# This script converts the net data to mat file

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb
MODEL1=bvlc_reference_rcnn_ilsvrc13
MODEL2=VGG_ILSVRC_19_layers

MODEL3=AlexNet_SalObjSub
#IMAGE1=cow
IMAGE2=child
IMAGE3=flower
IMGAE4=dog
echo "RUNNING $MODEL..."

python net2mat.py  $MODEL1 $IMAGE2
python net2mat.py  $MODEL1 $IMAGE3
python net2mat.py  $MODEL1 $IMAGE4

python net2mat.py  $MODEL2 $IMAGE2
python net2mat.py  $MODEL2 $IMAGE3
python net2mat.py  $MODEL2 $IMAGE4

python net2mat.py  $MODEL3 $IMAGE2
python net2mat.py  $MODEL3 $IMAGE3
python net2mat.py  $MODEL3 $IMAGE4




echo "Done."
