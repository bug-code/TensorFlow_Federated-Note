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


'''
生成处理后的手写数据集
'''
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


#函数预处理
def preprocess(dataset):
    #python函数嵌套，将dataset中的元素展平并返回
    def batch_format_fn(element):
        return collections.OrderedDict(x=tf.reshape(element['pixels'],[-1,784]) , 
                                    y=tf.reshape(element['label'] , [-1,1]))
    #将展平后的数据随机打乱，并组合成batch_size
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

#客户端
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
#生成联邦学习所需要使用的数据
#每个客户端的数据作为list中的一个元素，使用preprocess函数进行数据的预处理
federated_train_datasets = [preprocess(emnist_train.create_tf_dataset_for_client(x)) for x in sample_clients]
federated_test_datasets = [preprocess(emnist_test.create_tf_dataset_for_client(x)) for x in sample_clients]






'''
使用tff底层接口实现神经网络模型
'''
#定义一个存储结构，存储神经网络参数
#模型参数并累积统计信息
MnistVariables = collections.namedtuple('MnistVariables' , 'weights bias num_examples loss_sum accuracy_sum')
#定义神经网络参数
#参数全部使用tf.float32类型便于后续转换
#通过架构提供的源变量 ， 使用lambda表达式将变量进行封装和初始化
def create_mnist_variables( ):
    return MnistVariables(

        weights=tf.Variable(
            lambda : tf.zeros(dtype=tf.float32 , shape=(784,10)),
            name='weights',
            trainable = True
        ),

        bias = tf.Variable(
            lambda : tf.zeros(dtype=tf.float32 , shape=(10)),
            name='bias',
            trainable = True
        ),

        num_examples=tf.Variable(0.0 , name='num_example',trainable = False),

        loss_sum=tf.Variable(0.0 , name='loss_sum'  ,trainable=False),

        accuracy_sum=tf.Variable(0.0  , name='accuracy_sum' , trainable=False)
    )

#本地神经网络前向计算
def mnist_forward_pass(variables, batch):
    #前向矩阵输入与权重相乘并加偏置
    y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
    #反one-hot运算，获得预测输出
    predictions = tf.cast(tf.argmax(y, 1), tf.int32)
    #获取数据标签
    flat_labels = tf.reshape(batch['y'], [-1])
    #计算一个batch的平均loss值
    loss = -tf.reduce_mean(
        tf.reduce_sum(tf.one_hot(flat_labels,depth=10 , on_value=None , off_value = None) * tf.math.log(y), axis=[1]))
    #获取一个batch的平均准确率
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32))
    #获取样本数量
    num_examples = tf.cast(tf.size(batch['y']), tf.float32)
    #将样本数量添加至神经网络存储结构
    variables.num_examples.assign_add(num_examples)
    #累积一个batch的loss值总和
    variables.loss_sum.assign_add(loss * num_examples)
    #累积一个batch的准确率总和
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions

#返回本地指标（本地样本数量，平均loss值，平均准确率），用于评估和聚合到服务器
#联邦平均聚合算法，聚合平均准确率、loss值、样本数量
#用来衡量每个客户端的贡献
def get_local_mnist_metrics(variables):
    return collections.OrderedDict(
        num_examples = variables.num_examples,
        loss=variables.loss_sum / variables.num_examples , 
        accuracy = variables.accuracy_sum  / variables.num_examples
    )

#确定聚合策略，使用tff接口编写
#任何python函数，任何联邦学习的计算都可以使用@tff.federated_computation进行封装成tff类型
@tff.federated_computation
#metrics参数使用的是上述get_local_mnist_metrics()函数返回的矩阵值，
#该矩阵值不是tf张量，而是封装成tff张量，应当使用tff中的求平均值函数
def aggregate_mnist_metrics_accross_clients(metrics):
    #其中metrics是所有客户端中的样本数量，客户端训练后的平均准确率，客户端训练后的平均loss值
    return collections.OrderedDict(
        #获取样本数量
        num_examples = tff.federated_sum(metrics.num_examples),
        #计算平均loss值,以客户端样本数量作为权重
        loss = tff.federated_mean(metrics.loss , metrics.num_examples),
        #计算平均准确率
        accuracy = tff.federated_mean(metrics.accuracy , metrics.num_examples)
    )

#构建tff.learning.model模型接口
class MnistModel(tff.learning.Model):
    def __init__(self):
        #初始化神经网络参数矩阵
        self._variables = create_mnist_variables()
    #返回需要训练的神经网络权值和偏重
    @property
    def trainable_variables(self):
        return [self._variables.weights , self._variables.bias]
    
    #返回不需要训练的神经网络参数
    @property
    def non_trainable_variables(self):
        return []
    
    #返回本地参数 ， 样本数量，累计loss值 ， 累积准确率
    @property
    def local_variables(self):
        return [
            self._variables.num_examples , self._variables.loss_sum , self._variables.accuracy_sum
        ]
    
    #返回输入类型的spec值
    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None , 784] , tf.float32),
            y = tf.TensorSpec([None ,1] ,tf.int32)
        )

    #神经网络前向计算函数
    @tf.function
    def forward_pass(self , batch , training=True):
        del training
        loss , predictions = mnist_forward_pass(self._variables , batch)
        num_examples = tf.shape(batch['x'])[0]
        return tff.learning.BatchOutput(
            loss = loss , predictions = predictions , num_examples = num_examples
        )
    
    #返回本地输出参数值函数
    @tf.function
    def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)
    
    @property
    def federated_output_computation(self):
        #调用函数本体，而不是调用函数计算结果
        return aggregate_mnist_metrics_accross_clients
        




#创建迭代过程
#在服务器端由于不需要训练，所以只要指定本地客户端训练优化器
iterative_process = tff.learning.build_federated_averaging_process(
    #模型得是模型构造器而不是单个模型
    MnistModel , 
    client_optimizer_fn = lambda : tf.keras.optimizers.SGD(learning_rate=client_lr),
    server_optimizer_fn= lambda : tf.keras.optimizers.SGD(learning_rate=server_lr)
)
#迭代过程类初始化
#获取迭代状态
state = iterative_process.initialize()


logdir_for_compression = "/tmp/logs/scalars/lowlevel_model/"
summary_writer = tf.summary.create_file_writer(
    logdir_for_compression)

#基础训练测试
with summary_writer.as_default():
    for i  in range(NUM_ROUND):
        evaluation = tff.learning.build_federated_evaluation(MnistModel)
        pre_metrics = evaluation(state.model , federated_train_datasets)
        print('第', i , '轮训练前模型评估：',  pre_metrics , '\n')
        tf.summary.scalar('acc' , pre_metrics['accuracy' ],step=i)
        tf.summary.scalar('loss' , pre_metrics['loss' ],step=i)
        
        #第i轮模型训练
        state , metrics = iterative_process.next(state , federated_train_datasets)
        #第i轮模型训练后的测试，模型评估
        print('第', i , '轮模型训练参数：',  metrics)

        tf.summary.scalar('acc' , metrics['train']['accuracy' ],step=i)
        tf.summary.scalar('loss' , metrics['train']['loss' ],step=i)


        test_metrics = evaluation(state.model , federated_test_datasets)
        tf.summary.scalar('acc' , test_metrics['accuracy' ],step=i)
        tf.summary.scalar('loss' , test_metrics['loss' ],step=i)
        print('第', i , '轮模型测试评估：',  test_metrics , '\n')
        summary_writer.flush()

