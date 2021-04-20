import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import tensorflow as tf
from tensorflow_federated.python.simulation.models import mnist

#创建mnist神经网络模型
def mnist_model(comp_model = False):
    return mnist.create_keras_model(compile_model=comp_model)