import numpy as np
import matplotlib.pyplot as plt
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples，建议使用绝对路径
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import os
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
# 2、获取分类器并设定相关参数
# 通过下面命令获取训练模型
# ./scripts/download_model_binary.py models/bvlc_reference_caffenet
caffe.set_phase_test()
caffe.set_mode_cpu()
net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # 像素值范围[0,255]
net.set_channel_swap('data', (2,1,0))  # 训练模型是BGR而不是RGB,所以将测试图片转为BGR格式 
# 3、预测
## scores = net.predict([caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')])
scores = net.predict([caffe.io.load_image(caffe_root + 'examples/images/fish-bike.jpg')])
# 4、每一层的特征及大小
[(k, v.data.shape) for k, v in net.blobs.items()]
# 以('data', (10, 3, 227, 227))为例，‘data'表示层的名字，10表示批处理数据大小，3表示特征图的个数，227,227分别表示特征图的大小
# 5、每层参数及大小
[(k, v[0].data.shape) for k, v in net.params.items()]
# 以('conv1', (96, 3, 11, 11)为例，’conv1'表示层名，96表示滤波器个数，（3，11，11）表示滤波器大小，3为上一层feature map的个数，conv1的上一层是输入为RGB三个通道，因为feature map的个数为3。但对于('conv2', (256, 48, 5, 5))，上一层为 ('norm1', (10, 96, 27, 27)) feature map的个数为96，而48是92/2 ， 所以不太清楚是怎么实现的，猜测是第二个卷积层只从norm1层中选择一半进行卷积，可能得去具体研究一下模型了。
# 辅助函数：绘制特征图
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure() #新的绘图区
    plt.imshow(data)
# 8、显示输入图
plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))
plt.show()
# 9、"conv1"权重图
filters = net.params['conv1'][0].data  
vis_square(filters.transpose(0, 2, 3, 1)) 
plt.show()
# RGB转GBR可以看到是彩色图，因为每个滤波器有三个通道（3，10，10），总共96个。可以看到每个滤波器学到的是特征明显的边缘
# 10、显示”conv1"输出
feat = net.blobs['conv1'].data[4, :36]
vis_square(feat, padval=1)
plt.show()
# “conv1"的输出有256个feature map，这里只显示前36个，当然你也可以选择全部显示
# 12、可视化”conv2"的权重，“conv2"包含256个大小为 5*5*48的滤波器，这里只显示一部分
# 48**48 即 48*48。其实要观察第二层到底学习到什么特征，需要考虑第一层的权重，因为这是一个级联的过程，现在有一部分人已经做了这方面的工作了。
filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))
plt.show()
# 12、可视化”conv2"层的输出，即feature map
feat = net.blobs['conv2'].data[4, :36]
vis_square(feat, padval=1)
plt.show()
# 13、“conv3"层的feature map
feat = net.blobs['conv3'].data[4]
vis_square(feat, padval=0.5)
plt.show()
# 14、”conv4"层feature map
feat = net.blobs['conv4'].data[4]
vis_square(feat, padval=0.5)
plt.show()
# 同理可以观察你想输出的任意层的feature map
# 16、接下来看一下pooling层的影响 
# 下面是分别是"conv5" "pool5"的输出，可以看出通过pooling层后，每一个feature map的可区分性更强了，这正是分类模型所期望的
feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)
plt.show()
feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)
plt.show()
# 17、”fc6" "fc7"是两个全连接层，输出大小为4096*1，”fc6"层的分布比较均匀区分性比较弱，而通过“fc7"层各输出之间的可区分性增强
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()
# 18、“prob"层即预测层，预测该样本属于每一类的概率，ImageNet数据库有1000类，那么该层输出为1000*1
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
# 19、输出top 5的分类 
# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
## !../data/ilsvrc12/get_ilsvrc_aux.sh
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
print labels[top_k]