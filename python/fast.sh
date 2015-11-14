#!/usr/bin/env sh
# This script converts the net data to mat file

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb
MODEL1=bvlc_reference_rcnn_ilsvrc13
MODEL2=VGG_ILSVRC_19_layers
MODEL3=AlexNet_SalObjSub
MODEL4=bvlc_googlenet
IMAGE1="cow"
IMAGE2="child"
IMAGE3="flower"
IMGAE4="dog"
PYFILE="net2mat_paras_only.py"
#echo "RUNNING $MODEL..."

MODELLIST=[$MODEL4]
#IMAGELIST={"cow" "child" "flower" "dog"}
#IMAGELIST=IMAGE1 IMAGE2 IMAGE4 IMAGE3
for model in $MODELLIST
do
    for image in $IMAGELIST
    do
#	echo "runnig model $model on image $image jpg"
	python $PYFILE $model $image
    done
done



python $PYFILE  $MODEL4 $IMAGE2
python $PYFILE  $MODEL4 $IMAGE3
python $PYFILE  $MODEL4 $IMAGE4
python $PYFILE  $MODEL4 $IMAGE1

#python $PYFILE  $MODEL2 $IMAGE2
#python $PYFILE  $MODEL2 $IMAGE3
#python $PYFILE  $MODEL2 $IMAGE4

#python $PYFILE  $MODEL3 $IMAGE2
#python $PYFILE  $MODEL3 $IMAGE3
#python $PYFILE  $MODEL3 $IMAGE4




echo "Done."
