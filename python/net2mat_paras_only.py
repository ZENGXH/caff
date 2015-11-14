import caffe
import caffe.read_net
import caffe.get_net

import scipy.io as sio
import numpy as np
import re

def main(argv):
        MODEL = argv[1]
        image_name = argv[2]
        #image_name = 'cat'
        # MODEL = 'VGG'
       # layer_type = 'all'
        caffe_root = '../'

        layers = caffe.Read_net(MODEL)
        net_data = caffe.Get_net(MODEL,image_name)
        para = net_data.params
        parames = {}
        for keys,values in para.items():
          for i in range(0,len(values)):
            valSub = values[i]
            print 'collecting'+keys
            parames[keys+str(i)]=valSub.data
        matfile_para = caffe_root + 'python/_temp/'+MODEL+'_params.mat'

        print('parameters collect,',type(parames))
        sio.savemat(matfile_para,parames)
if __name__ == '__main__':
        import sys
        main(sys.argv)

