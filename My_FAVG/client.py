import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
# import numpy as np
import tensorflow as tf
from termcolor import colored
import random
from tensorflow_federated.python.simulation.models import mnist
from utils import *




'''
客户端类
'''
class client(object):
    def __init__(self ,
                local_dataset:dict, 
                client_name = 0 , 
                local_model = mnist.create_keras_model(compile_model=False)  
                 ):

        self.local_dataset = local_dataset
        self.client_name = client_name
        self.local_model = local_model
        self.dataset_size = get_datasize(self.local_dataset['train']) +get_datasize(self.local_dataset['test'])
        self.val_acc_list = []
        self.val_loss_list = []
        self.local_model_size_list = [ get_model_size(self.local_model) ]

    
    def set_client_name(self , name):
        self.client_name = name
    
    def set_model_weights(self , model : tf.keras.Model):
        self.local_model.set_weights(model.get_weights())
    
    
    def set_local_dataset(self , dataset):
        self.local_dataset = dataset


 
    
    def client_train(self , 
                    client_epochs = 10 , 
                    model_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True) ,
                    model_optimizer=tf.keras.optimizers.SGD(learning_rate=0.1) , 
                    model_metrics = ['accuracy']
                    ):
        self.local_model.compile(
                                optimizer=model_optimizer , 
                                loss=model_loss ,
                                metrics=model_metrics)
        client_train_history = self.local_model.fit(self.local_dataset['train'] 
                                                    , epochs = client_epochs 
                                                    , validation_data=self.local_dataset['test'] 
                                                    , validation_freq= client_epochs 
                                                    , verbose=0 
                                                    , workers= 4
                                                    , use_multiprocessing=True
                                                    )
        
        self.val_acc_list.append(client_train_history.history['val_accuracy'][0])
        self.val_loss_list.append(client_train_history.history['val_loss'][0])
        self.local_model_size_list.append(get_model_size(self.local_model))

    def get_local_info(self):
        return {'client_name': self.client_name , 
                'local_dataset_size': self.dataset_size , 
                'client_model_size_history': self.local_model_size_list ,
                'client_val_acc_history':self.val_acc_list , 
                'client_val_loss_history': self.val_loss_list ,
                'current_local_model_size': self.local_model_size_list[-1] , 
                'current_local_acc': self.val_acc_list[-1] , 
                'current_local_loss': self.val_loss_list[-1] 
                
                }


        