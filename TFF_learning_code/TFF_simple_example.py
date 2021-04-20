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
import time
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
nest_asyncio.apply()
tff.backends.reference.set_reference_context()

source, _ = tff.simulation.datasets.emnist.load_data()

#数据处理
def map_fn(example):
  return collections.OrderedDict(
      x=tf.reshape(example['pixels'], [-1, 784]), y=example['label'])


def client_data(n):
  ds = source.create_tf_dataset_for_client(source.client_ids[n])
  return ds.repeat(10).shuffle(500).batch(20).map(map_fn)


train_data = [client_data(n) for n in range(10)]
element_spec = train_data[0].element_spec

#模型创建函数
def model_fn():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(784,)),
      tf.keras.layers.Dense(units=10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])
  return tff.learning.from_keras_model(
      model,
      input_spec=element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#聚合算法，与后一个实验相比需要添加服务端优化策略 server_optimizer
trainer = tff.learning.build_federated_averaging_process(
    model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02))

'''
该处计算的只是训练准确率，而不是测试集上的准确率
'''
#模型评估函数
def evaluate(num_rounds=10):
  state = trainer.initialize()
  for _ in range(num_rounds):
    t1 = time.time()
    #在10个客户端上进行训练，并进行一次全局聚合
    state, metrics = trainer.next(state, train_data)
    t2 = time.time()
    # print(type(metrics))
    print('第',_ ,'轮准确率：',metrics['train']['sparse_categorical_accuracy'] , 'loss:' , metrics['train']['loss'])
    # print('metrics {m}, round time {t:.2f} seconds'.format(
    #     m=metrics, t=t2 - t1))

evaluate()
