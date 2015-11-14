#! /bin/bash
# brief: import models from caffe model zoo

FCN32S_PROTO_URL=https://gist.githubusercontent.com/longjon/ac410cad48a088710872/raw/fe76e342641ddb0defad95f6dc670ccc99c35a1f/fcn-32s-pascal-deploy.prototxt
FCN32=./fcn-32s-pascal/deploy.prototxt
#FCN8S_PROTO_URL=https://gist.githubusercontent.com/longjon/1bf3aa1e0b8e788d7e1d/raw/2711bb261ee4404faf2ddf5b9d0d2385ff3bcc3e/fcn-8s-pascal-deploy.prototxt
FCN8S_PROTO_URL=https://github.com/HyeonwooNoh/DeconvNet/blob/master/model/FCN/fcn-8s-pascal-deploy.prototxt
FCN8=./fcn-8s-pascal/deploy.prototxt
GPU=./places_googlenet


echo "downloading..."
#for model in ${MODELS_file[@]}; do
#    if [-f model];
#    then
#	echo "file $MODEL_FILE exist"
#    else
#	wget ${MODELS}
if [ ! -f  "$FCN32" ]
then 
    wget $FCN32S_PROTO_URL -O $FCN32
fi

if [ ! -f "$FCN8" ]
then
    wget $FCN8S_PROTO_URL -O $FCN8
fi

if [ ! -d "$GPU" ]
then 
    echo "no $GPU"
    mkdit $GPU
    cd $GPU
    wget http://vision.princeton.edu/pvt/GoogLeNet/Places/train_val_googlenet.prototxt -O deploy.prototxt
    wget http://vision.princeton.edu/pvt/GoogLeNet/Places/places_googlenet.caffemodel 
    cd ..
fi

#wget $FCN8S_PROTO_URL -O $FCN8

echo "done"
