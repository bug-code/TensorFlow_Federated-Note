from mnist import generate_clients_data
from client import client
from server import server
from mnist_model import mnist_model
from typing import List
import matplotlib.pyplot as plt
from utils import *
import numpy as np
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

    fig.tight_layout()
    plt.show()

def FAVG_init( ):
    # get IID data list for client list
    num_expamples_list_in_clients = [1000 , 2000 , 1500 , 500 , 3000 , 1000 , 1000,2000,1500 , 2000]
    client_train_data_list , client_test_data_list , server_test_dataset = generate_clients_data( num_expamples_list_in_clients , 
                                                                                                num_clients = 10, 
                                                                                                IsIID=True, 
                                                                                                batch_size=100,
                                                                                                tt_rate = 0.3)
    # experiment model
#     model = mnist_model(comp_model=False)

    # set dataset for server
    server_0 = server(test_dataset=server_test_dataset , server_model=mnist_model(comp_model=False)) 

    # client set list
    clients_list = []

    # set dataset for clients
    client_name_list = list('client_{}'.format(i) for i in range(10))
    for i in range(len(num_expamples_list_in_clients)):
        client_data_dict = {'train': client_train_data_list[i] , 'test' : client_test_data_list[i]  }
        clients_list.append( client(
                    local_dataset=client_data_dict, 
                    client_name = client_name_list[i] ,
                    local_model=  mnist_model(comp_model=False)
                    ) )
    return server_0 , clients_list
# train
def FAVG_train(server:server , clients_list: List[client] , server_round : int , client_enpochs:int):
    for i in range(server_round):
        for client_ in clients_list:
            client_.client_train(client_epochs=client_enpochs)
            # print(client_.get_local_info() , '\n')
        server.calculate_server_model(clients_list)
        server.broadcast_server_model(clients_list)
        server.server_model_test()
    
    return server , clients_list 

if __name__ == "__main__":
    server_0 , clients_list = FAVG_init()
    server_0 , clients_list = FAVG_train(server_0 , clients_list , 200 , 10)
    log_dir = "/tmp/logs/scalars/FAVG/"
    summary_acc_loss(logdir=log_dir , name=server_0 , loss=server_0.ave_loss_list , acc=server_0.ave_acc_list)
    # draw(server_0.ave_loss_list , server_0.ave_acc_list)
    np.savez('server_acc' , server_0.ave_acc_list)
    np.savez('server_loss' , server_0.ave_loss_list)
    for _client in clients_list:
        np.savez('{}_acc'.format(_client.client_name) , _client.val_acc_list )
        np.savez('{}_loss'.format(_client.client_name) , _client.val_loss_list )
            



