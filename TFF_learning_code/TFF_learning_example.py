import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import nest_asyncio
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

nest_asyncio.apply()


#导入手写数字数据集
emnist_train,emnist_test=tff.simulation.datasets.emnist.load_data()


#查看数据集大小
# print(len(emnist_train.client_ids))
#查看联邦学习数据集结构
print(emnist_train.element_type_structure)



#为客户端创建None-iid数据集
example_dataset=emnist_train.create_tf_dataset_for_client(
emnist_train.client_ids[0])



#获取第一个客户端的第一个数据
# example_element=next(iter(example_dataset))
#查看第一个客户端的第一个数据的类别
# print('\n',example_element['label'].numpy())
#数据图片显示
# plt.imshow(example_element['pixels'].numpy(),cmap='gray',aspect='equal')
# plt.grid(False)
# plt.show()





#显示第一个客户端中的前40张图片
# figure = plt.figure(figsize=(20,4))
# j = 0

# for example in example_dataset.take(40):
#     plt.subplot(4,10,j+1)
#     plt.imshow(example['pixels'].numpy(),cmap='gray',aspect='equal')
#     plt.axis('off')
#     plt.grid(False)
#     j+=1
# plt.show()




#显示每个客户端的数据统计
# fig = plt.figure(figsize=(12,7))
# fig.suptitle('每个客户端的样本标签数量统计')
# for i in range(6):
#     #获取第i个客户端的数据
#     client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
#     #获取数据集分布情况字典
#     #plot_data = collections.defaultdict(list)
#     for example in client_dataset:
#         label  =example['label'].numpy()
#         plot_data[label].append(label)
#         plt.subplot(2,3,i+1)
#         plt.title('客户端{}'.format(i))
#         for j in range(10):
#             plt.hist(plot_data[j] , density=False , bins=[0,1,2,3,4,5,6,7,8,9,10])
# plt.show()




# 可视化每个客户端的像素平均图像 ， 每个客户端的风格不一样，图像的像素平均值不同
# for i in range(5):
#     client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
#     plot_data = collections.defaultdict(list)
#     for example in client_dataset:
#         plot_data[example['label'].numpy()].append(example['pixels'].numpy())
#     f = plt.figure(i , figsize=(12,5))
#     f.suptitle('客户端{}平均像素图片'.format(i))
#     for j in range(10):
#         mean_img = np.mean(plot_data[j] , 0)
#         plt.subplot(2,5,j+1)
#         plt.imshow(mean_img.reshape((28,28)))
#         plt.axis('off')
# plt.show()


#预处理输入数据
NUM_CLIENTS=10
NUM_EPOCHS=5
NUM_ROUND = 10
BATCH_SIZE=20
SHUFFLE_BUFFER=100
PREFETCH_BUFFER=10

def preprocess(dataset):
    #python函数嵌套，将dataset中的元素展平并返回
    def batch_format_fn(element):
        return collections.OrderedDict(x=tf.reshape(element['pixels'],[-1,784]) , y=tf.reshape(element['label'] , [-1,1]))
    #将展平后的数据随机打乱，并组合成batch_size
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

preprocessed_example_dataset=preprocess(example_dataset)
# #next函数，不断取出其中可迭代对象的所有元素，tf.map_structure函数，将取出的函数进行前面的lambda运算
sample_batch=tf.nest.map_structure(lambda x : x.numpy(),next(iter(preprocessed_example_dataset)))
# print(sample_batch)


#生成联邦学习所需要使用的数据
def make_federated_data(client_data , client_ids):
    #每个客户端的数据作为list中的一个元素，使用preprocess函数进行数据的预处理
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

#客户端
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train , sample_clients)

print('客户端数量；{l}'.format(l=len(federated_train_data)))

print('\n' , '第一个数据集：{d}'.format(d = federated_train_data[0]))

#创建神经网络模型
def create_keras_model( ):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10 , kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


#keras创建的神经网络模型需要封装在tff.learning.from_keras_model接口中
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model , 
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

#训练过程通过使用联邦平均算法，tff.learning.buid_federared_averaging_process函数
#其中第一个参数必须是一个神经网络模型的构造器，而不是构建完成的神经网络，
# 因为在训练过程中无论是客户端还是服务器端，其神经网络的参数都是随着训练的进行更新的，
# 而不是使用一个构建完成的神经网络模型。

iterative_process  = tff.learning.build_federated_averaging_process(
    model_fn,
    #客户端优化器，只针对客户端本地模型进行更新优化
    client_optimizer_fn=lambda : tf.keras.optimizers.SGD(learning_rate=0.02),
    #服务器端优化器，只针对服务器端全局模型进行更新优化
    server_optimizer_fn=lambda : tf.keras.optimizers.SGD(learning_rate=1.0)
)

# print(str(iterative_process.initialize.type_signature))

#模型初始化，设置神经网络模型，客户端优化函数，服务器优化函数
state = iterative_process.initialize()

# print('\n\n',state)
#神经网络模型训练10轮情况 （还未添加客户端）
# for i  in range(NUM_ROUND):
#     state , metrics = iterative_process.next(state , federated_train_data)
#     print('第', i , '轮模型参数{}'.format(metrics))

##可视化网络模型训练结果
#创建目录及摘要写入器，写入训练日志

logdir = "/tmp/logs/scakars/training/"
summary_writer = tf.summary.create_file_writer(logdir)

#绘制训练日志
with summary_writer.as_default():
    for round_num in range(NUM_ROUND):
        state , metrics = iterative_process.next(state , federated_train_data)
        for name , value in metrics['train'].items():
            tf.summary.scalar(name , value , step=round_num)