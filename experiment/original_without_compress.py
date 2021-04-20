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

'''
原始联邦学习平均算法，显示客户端需要上传的数据量大小，显示服务端需要广播的数据量大小
计算训练的准确率，loss值，在tensorboard中显示训练结果
'''

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



# This value only applies to EMNIST dataset, consider choosing appropriate
# values if switching to other datasets.


def reshape_emnist_element(element):
  return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

def preprocess_train_dataset(dataset):
  """Preprocessing function for the EMNIST training dataset."""
  return (dataset
          # Shuffle according to the largest client dataset
          .shuffle(buffer_size=MAX_CLIENT_DATASET_SIZE)
          # Repeat to do multiple local epochs
          .repeat(CLIENT_EPOCHS_PER_ROUND)
          # Batch to a fixed client batch size
          .batch(CLIENT_BATCH_SIZE, drop_remainder=False)
          # Preprocessing step
          .map(reshape_emnist_element))





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



def tff_model_fn():
  keras_model = create_original_fedavg_cnn_model()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])








def format_size(size):
  """A helper function for creating a human-readable size."""
  size = float(size)
  for unit in ['bit','Kibit','Mibit','Gibit']:
    if size < 1024.0:
      return "{size:3.2f}{unit}".format(size=size, unit=unit)
    size /= 1024.0
  return "{size:.2f}{unit}".format(size=size, unit='TiB')

def set_sizing_environment():
  """Creates an environment that contains sizing information."""
  # Creates a sizing executor factory to output communication cost
  # after the training finishes. Note that sizing executor only provides an
  # estimate (not exact) of communication cost, and doesn't capture cases like
  # compression of over-the-wire representations. However, it's perfect for
  # demonstrating the effect of compression in this tutorial.
  sizing_factory = tff.framework.sizing_executor_factory()

  # TFF has a modular runtime you can configure yourself for various
  # environments and purposes, and this example just shows how to configure one
  # part of it to report the size of things.
  context = tff.framework.ExecutionContext(executor_fn=sizing_factory)
  tff.framework.set_default_context(context)

  return sizing_factory


def train(federated_averaging_process, num_rounds, num_clients_per_round, summary_writer):
  """Trains the federated averaging process and output metrics."""
  # Create a environment to get communication cost.
  environment = set_sizing_environment()

  # Initialize the Federated Averaging algorithm to get the initial server state.
  state = federated_averaging_process.initialize()

  with summary_writer.as_default():
    for round_num in range(num_rounds):
      # Sample the clients parcitipated in this round.
      sampled_clients = np.random.choice(
          emnist_train.client_ids,
          size=num_clients_per_round,
          replace=False)
      # Create a list of `tf.Dataset` instances from the data of sampled clients.
      sampled_train_data = [
          emnist_train.create_tf_dataset_for_client(client)
          for client in sampled_clients
      ]
      # Round one round of the algorithm based on the server state and client data
      # and output the new state and metrics.
      state, metrics = federated_averaging_process.next(state, sampled_train_data)

      # For more about size_info, please see https://www.tensorflow.org/federated/api_docs/python/tff/framework/SizeInfo
      size_info = environment.get_size_info()
      broadcasted_bits = size_info.broadcast_bits[-1]
      aggregated_bits = size_info.aggregate_bits[-1]

      print('round {:2d}, metrics={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, metrics, format_size(broadcasted_bits), format_size(aggregated_bits)))

      # Add metrics to Tensorboard.
      for name, value in metrics['train'].items():
          tf.summary.scalar(name, value, step=round_num)

      # Add broadcasted and aggregated data size to Tensorboard.
      tf.summary.scalar('cumulative_broadcasted_bits', broadcasted_bits, step=round_num)
      tf.summary.scalar('cumulative_aggregated_bits', aggregated_bits, step=round_num)
      summary_writer.flush()


MAX_CLIENT_DATASET_SIZE = 418

CLIENT_EPOCHS_PER_ROUND = 1
CLIENT_BATCH_SIZE = 20
TEST_BATCH_SIZE = 500

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
    only_digits=True)

emnist_train = emnist_train.preprocess(preprocess_train_dataset)

# Gets the type information of the input data. TFF is a strongly typed
# functional programming framework, and needs type information about inputs to 
# the model.
input_spec = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0]).element_spec

federated_averaging = tff.learning.build_federated_averaging_process(
    model_fn=tff_model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

logdir = "/tmp/logs/scalars/original/"
summary_writer = tf.summary.create_file_writer(logdir)

train(federated_averaging_process=federated_averaging, num_rounds=10,
      num_clients_per_round=10, summary_writer=summary_writer)
