import tensorflow as tf
# import numpy as np
from tensorflow_federated.python.simulation.models import mnist
from client import client
from typing import List
from utils import *


type_client_list = List[client]

class server(object):
    def __init__(self , 
                server_name = 0 ,
                test_dataset = None ,
                server_model = mnist.create_keras_model(compile_model=False) ):
        self.server_name = server_name
        self.test_dataset = test_dataset
        self.server_model = server_model
        self.ave_acc_list = []
        self.ave_loss_list = []
    
    def get_server_info(self):
         return {
             'server name' : self.server_name , 
             'server dataset size' : get_datasize(self.test_dataset) , 
             'server model size' : get_model_size(self.server_model) , 
             'server acc history' : self.ave_acc_list , 
             'server loss history' : self.ave_loss_list
         }

    #calculate server model by clients list
    def calculate_server_model( self, client_list : type_client_list):
        # get sum of datasets size in clients
        sum_client_datasets = 0
        for client in client_list:
            sum_client_datasets +=client.dataset_size
        # get client impact factor to server model
        rate_client_dataset_size = []
        for client in client_list:
            rate_client_dataset_size.append( client.local_model_size_list[-1] /sum_client_datasets )
        #calculate server model
        clients_modelweight_list = []
        for (client , factor) in zip(client_list,rate_client_dataset_size):
            client_union_weights = []
            client_weights = client.local_model.get_weights()
            num_client_layers = len(client_weights)
            for i in range(num_client_layers):
                client_union_weights.append(factor*client_weights[i])
            clients_modelweight_list.append(client_union_weights)
        # union server model
        metrix = []
        for weights in zip(*clients_modelweight_list):
            weights_sum = tf.reduce_sum(weights, axis =0)
            metrix.append(weights_sum)
        self.server_model.set_weights(metrix)
            

    #broadcast server model to clients list
    def broadcast_server_model( self, client_list : type_client_list ):
        for client in client_list:
            client.set_model_weights(self.server_model)
        # return client_list
    
    # server test
    def server_model_test(self ):
        server_loss , server_acc =self.server_model.evaluate(self.test_dataset , verbose=1 ,  workers=4 , use_multiprocessing=True )
        self.ave_loss_list.append(server_loss)
        self.ave_acc_list.append(server_acc)
