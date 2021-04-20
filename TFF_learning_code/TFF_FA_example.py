import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import layers , Sequential , losses , metrics , optimizers , datasets
import nest_asyncio
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
nest_asyncio.apply()
tff.backends.reference.set_reference_context()

'''
低层级的联邦学习平均算法：
    使用keras API和tff.simulation实现
'''


'''
数据处理
'''
#准备联邦数据集：Non-IID类型数据
#假设场景 有十个客户端，每个客户端贡献自己独有的信息进行联邦学习训练

#加载Mnist数据集
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()


#数据查看
# for i , d in  enumerate( mnist_train[1] ):
#   if i <10:
#     print(i , d , '\n')


#显示联邦学习数据类型和形状
#6万张图片，每张图片像素为28*28 ， 张量类型（60000 ， 28 ， 28）
#6万张图片标签 （60000，）
#以上都在单张图片中的第一维度
# print([(x.dtype , x.shape) for x in mnist_train])

#数据预处理函数
#将原始Mnist数据转换成联邦序列数据，以便联邦学习计算
#mnist数据集中拥有70000张图片，其中60000张用于训练集，10000张用于测试集

#设置每个客户端中样本数量
#在该示例中，所有客户端共拥有10000张图片 ，
#每个客户端1000张，测试集共10000张图片，也就是说没有取所有的数据进行训练
NUM_EXAMPLES_PER_USER = 1000
#设置客户端训练时的batch_size,也就是说客户端一个epoch中一次训练100张图片，10次训练完
BATCH_SIZE = 100


def get_data_for_digit(source, digit):
  output_sequence = []
  #取出相同数据标签类别的数据索引，作为一个客户端的数据，
  # 比如客户端0所拥有的数据类别都是0 ，以此类推
  all_samples = [i for i, d in enumerate(source[1]) if d == digit]
  #将客户端数据切分成batch_size==100
  for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
    batch_samples = all_samples[i:i + BATCH_SIZE]
    output_sequence.append({
        'x':
            #将该batch_size中的数据展平，并进行（0，1）标准化
            np.array([source[0][i].flatten()/255.0 for i in batch_samples],
                     dtype=np.float32),
        'y':
            #取出该batch_size中数据对应的标签
            np.array([source[1][i] for i in batch_samples], dtype=np.int32)
    })
  return output_sequence

federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]
#查看数据
# print(len(federated_train_data[5][-1]['y']))
# plt.imshow(federated_train_data[5][-1]['x'][-1].reshape(28, 28), cmap='gray')
# plt.grid(False)
# plt.show()

'''使用独立的python函数编写复杂的TF逻辑代码，不推荐使用tff.tf_computation装饰器
    自定义损失函数
'''
#定义TFF命名元组
#定义联邦训练时的输入类型
BATCH_SPEC = collections.OrderedDict(
  #因为一个batch中数据的数量不确定，所以定义为未知
  x = tf.TensorSpec(shape=[None , 784] , dtype=tf.float32) , 
  #由于标签类型在内存中存储的是int类型，所以需要定义成int类型，否则会报错
  y = tf.TensorSpec(shape=[None] , dtype=tf.int32)
)

#训练时batch的输入类型
BATCH_TYPE = tff.to_type(BATCH_SPEC)
# print(BATCH_TYPE)

#TFF有单独的抽象类型构造器tff.StructType ， 
#由于TFF底层的代码不是python,所以不能使用python类型直接进行计算
#使用python类型需要对其进行tff装饰，装饰成tff可以使用的类型

#获取神经网络模型的参数规格
MODEL_SPEC = collections.OrderedDict(
  #10表示输入层有10个神经元
  weights = tf.TensorSpec(shape=[784 , 10],dtype=tf.float32),
  bias = tf.TensorSpec(shape=[10] , dtype=tf.float32)
)

MODEL_TYPE = tff.to_type(MODEL_SPEC)
# print(MODEL_TYPE)

#在装饰器tff.tf_computation中使用tf.function装饰器 ，
#则可以在其内部使用python语言编写代码逻辑
#tff.tf_computation 函数可以调用tf.function装饰器 装饰的代码逻辑
#但是反之不能

@tf.function
def forward_pass(model , batch):
  # CC_loss = losses.CategoricalCrossentropy(from_logits=True)
  pre_y = tf.nn.softmax(
    tf.matmul(batch['x'], model['weights']) + model['bias']
  )
  return  -tf.reduce_mean(
    # CC_loss(tf.one_hot(batch['y'], depth = 10 , on_value = None) , pre_y)
      tf.reduce_sum(
          tf.one_hot(batch['y'], 10  , on_value=None , off_value=None ) * tf.math.log(pre_y), axis=[1])
    )

@tff.tf_computation(MODEL_TYPE , BATCH_TYPE)
def batch_loss(model , batch):
  return forward_pass(model , batch)

# print(str(batch_loss.type_signature))
#形式上TFF计算只接收单个参数 ， 但是如果参数类型是原则，则可以使用熟悉的python语法
#例如下面这个初始化模型一样
initial_model = collections.OrderedDict(
  weights  = np.zeros([784,10] , dtype=np.float32),
  bias = np.zeros([10] , dtype=np.float32)
)
#标签为5的最后一个batch数据集进行测试
# client_5_last_batch = federated_train_data[5][-1]

# print( batch_loss(initial_model , client_5_last_batch) )



'''
单个batch中的梯度下降算法
'''
#由于序列化会丢失一些调试信息，同时难以调试，所以尽量不要使用tff.tf_computation装饰器来 测试 tf代码
#tff.tf_computation装饰的函数可以内联在tff.tf_computation装饰的函数中，但是不建议这样做。
#在tff.tf_computation装饰的函数中最好将内部函数设置为常规的python函数或者tff函数
#如果需要调用的函数是tff类型的，则要使用tff装饰
@tff.tf_computation(MODEL_TYPE , BATCH_TYPE , tf.float32)
def batch_train(initial_model , batch , lr):
  model_vars = collections.OrderedDict([
    (name , tf.Variable(name=name , initial_value=value)) 
    for name , value in initial_model.items()
  ])
  optimizer = optimizers.SGD(learning_rate= lr)
  @tf.function
  def _train_on_batch(model_vars , batch):
    #通过由上述函数求得的loss来对模型进行梯度下降
    with tf.GradientTape() as tape:
      loss = forward_pass(model_vars , batch)
    grads = tape.gradient(loss , model_vars)
    optimizer.apply_gradients(zip(tf.nest.flatten(grads) , tf.nest.flatten(model_vars) ))
    return model_vars
  return _train_on_batch(model_vars , batch)

# print(str(batch_train.type_signature))

# #梯度下降测试
# model = initial_model
# sum_loss = []
# for _ in range(5):
#   model = batch_train(model , client_5_last_batch , 0.01)
#   sum_loss.append(batch_loss(model , client_5_last_batch))

# print(sum_loss)

'''
梯度下降在本地数据中(本地客户端的所有batch数据)
所有本地数据的梯度下降方法输入的数据类型为：ttf.SequenceType(BATCH_TYPE)
'''
#这种方式实现的客户端梯度下降方法所使用的学习率都是一样的

#定义本地客户端输入所有数据的数据类型
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

#定义该函数的数据输入类型
@tff.federated_computation(MODEL_TYPE , tf.float32 , LOCAL_DATA_TYPE)
def local_train(initial_model , lr , all_batchs):
  #每个batch数据都使用MAP函数进行计算
  @tff.federated_computation(MODEL_TYPE , BATCH_TYPE)
  def batch_fn(model , batch):
    #由于batch_train函数接收的是三个参数，而tff.sequence_reduce处理的是两个参数
    #因此将batch_train嵌入到该函数中，由外函数提供学习率lr
    return batch_train(model , batch , lr)
  #模型训练，将所有客户端数据，逐个batch的调用上述的batch_train函数进行训练
  #使用的是batch_train中的SGD梯度下降方法，
  # 当对所有batch数据训练完成，也就对整个客户端数据训练完成
  #tff.sequence_reduce函数应用于联邦计算函数中，tff.sequence_reduce内不能包含tf代码
  return tff.sequence_reduce(all_batchs , initial_model , batch_fn)

# print(local_train.type_signature)



'''
本地模型评估
'''
#在评估本地模型时，有两种策略
#1、计算本地所有batch数据的总loss值，与初始模型状态对比看是否下降
#2、计算本地所有batch数据的平均loss值，与初始模型状态相比看是否下降
@tff.federated_computation(MODEL_TYPE , LOCAL_DATA_TYPE)
def local_eval(model , all_batchs):
  #计算平均loss值时使用tff.sequence_average函数
  return tff.sequence_sum(
    #序列map函数，将all_batchs中的每个batch都进行loss值计算
    #tff.sequence_map与tff.sequence_reduce的差别在于map是并行计算，reduce是串行计算
    tff.sequence_map(
      tff.federated_computation(lambda b: batch_loss(model , b) ,BATCH_TYPE) , 
      all_batchs
    )
  )

# print(local_eval.type_signature)

# #评估初始模型在客户端5上的表现

# #在客户端5中训练所有数据
client_5_batchs = federated_train_data[5]
client_5_train_model = local_train(initial_model , 0.01 , client_5_batchs)
# print('init model on client_5_datasets loss = ' , local_eval(initial_model , client_5_batchs) , '\n')
# print('client5 model  on client_5_datasets loss = ' , local_eval(client_5_train_model , client_5_batchs) , '\n')

# client_0_batchs = federated_train_data[0]
# # client_0_train_model = local_train(initial_model , 0.01 , client_5_batchs)
# # print('init model on client_0_datasets loss = ' , local_eval(initial_model , client_0_batchs) , '\n')
# print('client5 model on client_0_datasets loss = ' , local_eval(client_5_train_model , client_0_batchs) , '\n')



'''
联邦学习评估：
  1、定义服务器端的模型类型
  2、定义客户端的数据类型
'''
SERVER_MODEL_TYPE = tff.type_at_server(MODEL_TYPE)
CLIENT_DATA_TYPE = tff.type_at_clients(LOCAL_DATA_TYPE)

#将聚合的模型分发给所有的被选中的客户端,
#所有客户端接收到全局模型后，使用本地数据进行计算，求取本地的平均损失率，用于评估联邦模型的效果
#模型服务端发送到客户端
#平均loss值客户端发送到服务端
@tff.federated_computation(SERVER_MODEL_TYPE , CLIENT_DATA_TYPE)
def federated_eval(model , data):
  return tff.federated_mean(
    #[]是隐式转换到tff类型
    tff.federated_map(local_eval , [tff.federated_broadcast(model) , data] )
  )
#在tff中存在隐式转换，例如{<X,Y>}@Z 等价于 <{X}@Z , {Y}@Z>
# print(federated_eval.type_signature)

#计算上述在客户端5中产生的模型其他所有客户端中的loss值 ，
#求所有客户端的平均loss值作为衡量联邦学习效果
# print('initial model federated loss = ',federated_eval(initial_model , federated_train_data),'\n')
# print('client_5_train_model federated loss = ' , federated_eval(client_5_train_model , federated_train_data),'\n')

'''
联邦学习训练：
  最简单的联邦学习训练方法
'''
#将tf.float32类型封装成tff中的float32类型
SERVER_FLOAT_TYPE = tff.type_at_server(tf.float32)
@tff.federated_computation(SERVER_MODEL_TYPE , SERVER_FLOAT_TYPE , CLIENT_DATA_TYPE)
def federated_train(model , lr ,data):
  #返回的是训练后的模型
  return tff.federated_mean(
    tff.federated_map(local_train , [
      tff.federated_broadcast(model) , 
      tff.federated_broadcast(lr),
      data
    ])
  )
  
# print(federated_train.type_signature)

logdir_for_custom_FA = "/tmp/logs/scalars/custom_FA/"
summary_writer_for_custom_FA = tf.summary.create_file_writer(logdir_for_custom_FA)

#联邦训练测试
model = initial_model
lr = 0.1
NUM_ROUNDS = 200
with summary_writer_for_custom_FA.as_default():  
  for round in range(NUM_ROUNDS):
    model = federated_train(model, lr , federated_train_data)
    lr = lr*0.9
    loss = federated_eval(model , federated_train_data)
    tf.summary.scalar('loss' , loss , step=round)
    print('round {} , loss = {}'.format(round , loss))
    summary_writer_for_custom_FA.flush()

#查看模型在测试集上的效果
print('initial_model test loss =',
      federated_eval(initial_model, federated_test_data))
print('trained_model test loss =', federated_eval(model, federated_test_data))
