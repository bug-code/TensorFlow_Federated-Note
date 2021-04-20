import numpy as np
import tensorflow as tf

def preprocess_client_data(data, bs=100):
    x, y = zip(*data) 

    return tf.data.Dataset.from_tensor_slices( ( list(x) , list(y) ) ).batch(bs)

'''
shuffle dataset
'''
def shuffle_dataset(datas , labels):
    shuffle_ix = np.random.permutation(np.arange(len(datas)))

    return datas[shuffle_ix] , labels[shuffle_ix]

'''
获取客户端模型大小
'''
def get_model_size(model):
	para_num = sum([np.prod(w.shape) for w in model.get_weights()])
	# para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
	para_size = para_num * 4 / 1024 / 1024
	return para_size

'''
获取客户端数据集大小
'''

def get_datasize(dataset):
    dataset_size = 0
    for batch in dataset:
        dataset_size += len(batch)
    
    return dataset_size


def summary_acc_loss(logdir , name ,loss:list ,acc:list ):
    summary_writer = tf.summary.create_file_writer(logdir)
    for rnd in range(len(loss)):            
        tf.summary.scalar('{}_acc'.format(name) , acc[rnd] , step=rnd)
        tf.summary.scalar('{}_loss'.format(name) , loss[rnd] , step=rnd)
        summary_writer.flush()