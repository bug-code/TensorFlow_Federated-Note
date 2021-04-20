import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import nest_asyncio
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
nest_asyncio.apply()



NUM_CLIENTS=10
NUM_EPOCHS=5
NUM_ROUND = 200
BATCH_SIZE=20
SHUFFLE_BUFFER=100
PREFETCH_BUFFER=10

client_lr =0.02
server_lr = 1.0
#导入手写数字数据集
emnist_train,emnist_test=tff.simulation.datasets.emnist.load_data()

'''
生成处理后的手写数据集
'''

#函数预处理
def preprocess(dataset):
    #python函数嵌套，将dataset中的元素展平并返回
    def batch_format_fn(element):
        return collections.OrderedDict(x=tf.reshape(element['pixels'],[-1,784]) , 
                                    y=tf.reshape(element['label'] , [-1,1]))
    #将展平后的数据随机打乱，并组合成batch_size
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_datasets =[preprocess(emnist_train.create_tf_dataset_for_client(x)) for x in sample_clients]

federated_test_datasets =[preprocess(emnist_test.create_tf_dataset_for_client(x)) for x in sample_clients] 

input_spec = federated_train_datasets[0].element_spec    

#神经网络模型
def create_keras_model( ):
    return  tf.keras.models.Sequential([
                                            tf.keras.layers.Input(shape=(784,)),
                                            tf.keras.layers.Dense(10 , kernel_initializer='zeros'),
                                            tf.keras.layers.Softmax(),
                                        ])

#将提供的神经网络模型封装在tff.learning.from_keras_model接口中
def model_fn():
    model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model =model, 
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

fed_aver = tff.learning.build_federated_averaging_process(
                        model_fn,
                        #客户端优化器，只针对客户端本地模型进行更新优化
                        client_optimizer_fn=lambda : tf.keras.optimizers.SGD(learning_rate=client_lr),
                        #服务器端优化器，只针对服务器端全局模型进行更新优化
                        server_optimizer_fn=lambda : tf.keras.optimizers.SGD(learning_rate=server_lr)
                    )

state = fed_aver.initialize()

logdir_for_compression = "/tmp/logs/scalars/custom_model/"
summary_writer = tf.summary.create_file_writer(
    logdir_for_compression)

with summary_writer.as_default():    
    #基础训练测试
    for i  in range(NUM_ROUND):
        state , metrics = fed_aver.next(state , federated_train_datasets)
        test_state , test_metrics = fed_aver.next(state , federated_test_datasets)
        print('第', i , '轮训练模型loss：',  metrics['train']['loss'] , '准确率：', metrics['train']['sparse_categorical_accuracy'] , '\n')
        tf.summary.scalar('train_loss',metrics['train']['loss'], step=i)
        tf.summary.scalar('train_acc',metrics['train']['sparse_categorical_accuracy'], step=i)
        print('第', i , '轮测试模型loss：',  test_metrics['train']['loss'] , '准确率：', test_metrics['train']['sparse_categorical_accuracy'] , '\n')
        tf.summary.scalar('test_loss',test_metrics['train']['loss'], step=i)
        tf.summary.scalar('test_acc',test_metrics['train']['sparse_categorical_accuracy'], step=i)
        summary_writer.flush()