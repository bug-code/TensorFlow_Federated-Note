import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import nest_asyncio
import tensorflow as tf
import tensorflow_federated as tff
import functools
import time
import numpy as np
import matplotlib.pyplot as plt
nest_asyncio.apply()

#画图函数
def draw(epoch_sumloss , epoch_acc):
    x=[i for i in range(len(epoch_sumloss))]
    #左纵坐标
    fig , ax1 = plt.subplots()
    color = 'red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss' , color=color)
    ax1.plot(x , epoch_sumloss , color=color)
    ax1.tick_params(axis='y', labelcolor= color)

    ax2=ax1.twinx()
    color1='blue'
    ax2.set_ylabel('acc',color=color1)
    ax2.plot(x , epoch_acc , color=color1)
    ax2.tick_params(axis='y' , labelcolor=color1)


    plt.legend()
    fig.tight_layout()
    plt.show()



'''
最原始的联邦学习聚合算法实现
'''



'''
数据预处理模块
'''
# This value only applies to EMNIST dataset, consider choosing appropriate
# values if switching to other datasets.
MAX_CLIENT_DATASET_SIZE = 418

CLIENT_EPOCHS_PER_ROUND = 1
CLIENT_BATCH_SIZE = 20
TEST_BATCH_SIZE = 500

def reshape_emnist_element(element):
  return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

def preprocess_train_dataset(dataset):
  """Preprocessing function for the EMNIST training dataset."""
  return (dataset
          # Shuffle according to the largest client dataset
          .shuffle(buffer_size=MAX_CLIENT_DATASET_SIZE)
          # Repeat to do multiple local epochs
          #要实现论文聚合策略的算法，客户端的训练次数需要在此进行改进        
          .repeat(CLIENT_EPOCHS_PER_ROUND)
          # Batch to a fixed client batch size
          .batch(CLIENT_BATCH_SIZE, drop_remainder=False)
          # Preprocessing step
          .map(reshape_emnist_element))


emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
    only_digits=True)
#预处理后的训练集
emnist_train = emnist_train.preprocess(preprocess_train_dataset)
# print(type(emnist_train))
#预处理后的测试集
emnist_test = emnist_test.preprocess(preprocess_train_dataset)





'''
卷积神经网络模型构建模块
'''
def create_original_fedavg_cnn_model(only_digits=True):
  """The CNN model used in https://arxiv.org/abs/1602.05629."""
  data_format = 'channels_last'

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
      conv2d(filters=32),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10 if only_digits else 62),
      tf.keras.layers.Softmax(),
  ])

  return model

# Gets the type information of the input data. TFF is a strongly typed
# functional programming framework, and needs type information about inputs to 
# the model.
input_spec = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]).element_spec

def tff_model_fn():
  keras_model = create_original_fedavg_cnn_model()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])





'''
联邦平均聚合算法：包含模型构造器 ， 客户端优化功能，服务端优化功能
'''
federated_averaging = tff.learning.build_federated_averaging_process(
    model_fn=tff_model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


'''
模型测试
'''
#模型评估函数
def evaluate(num_rounds=10  , num_clients_per_round = 10):
    state = federated_averaging.initialize()
    train_acc_loss = []
    test_acc_loss = []
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for _ in range(num_rounds):
        #从客户端集合中选择若干客户端参与训练
        sampled_clients = np.random.choice(
            emnist_train.client_ids , 
            size=num_clients_per_round,
            replace=False
        )
        #为选择的客户端创建数据集
        sampled_train_data = [
            emnist_train.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        sampled_test_data = [
            emnist_test.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]

        t1 = time.time()
        #在10个客户端上进行训练，并进行一次全局聚合
        state, metrics = federated_averaging.next(state, sampled_train_data)
        train_acc.append(metrics['train']['sparse_categorical_accuracy'])
        train_loss.append(metrics['train']['loss'])

        t2 = time.time()
        #测试模型在测试集上的效果
        test_state , test_metrics = federated_averaging.next(state , sampled_test_data)
        test_acc.append(test_metrics['train']['sparse_categorical_accuracy'])
        test_loss.append(test_metrics['train']['loss'])

        # print(type(metrics))
        # print('第',_ ,'轮准确率：',test_metrics['train']['sparse_categorical_accuracy'] , 'loss:' , test_metrics['train']['loss']  ,'time:' , t2-t1 , '\n')
        # print('metrics {m}, round time {t:.2f} seconds'.format(
        #     m=metrics, t=t2 - t1))
    train_acc_loss.append(train_acc)
    train_acc_loss.append(train_loss)
    test_acc_loss.append(test_acc)
    test_acc_loss.append(test_loss)

    return train_acc_loss , test_acc_loss


train_acc_loss , test_acc_loss  = evaluate( )
draw(train_acc_loss[1] , train_acc_loss[0])
draw(test_acc_loss[1] , test_acc_loss[0])
