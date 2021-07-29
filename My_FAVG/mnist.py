import random
import numpy as np
from termcolor import colored
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from mnist_model import mnist_model
from utils import *
def generate_clients_data( num_expamples_list_in_clients:list , num_clients = 10, IsIID=True, batch_size=100 ,tt_rate = 0.3):
        
    (x_train, y_train), (x_test, y_test)= mnist.load_data()
    x_train , y_train = shuffle_dataset(x_train , y_train)
    x_test , y_test = shuffle_dataset(x_test  , y_test)

    x_train = x_train.astype('float32').reshape(-1,28*28)/255.0
    x_test = x_test.astype('float32').reshape(-1,28*28)/255.0
    y_test = tf.one_hot(y_test , depth=10 , on_value=None , off_value = None)
    y_train = tf.one_hot(y_train , depth=10 , on_value=None , off_value = None)



    
    if len(num_expamples_list_in_clients) == 1:
        num_expamples_list_in_clients *= num_clients 
    
    # dataset for server test 
    # get server datasets 
    client_dataset_test_size  = int(sum([x*tt_rate for x in num_expamples_list_in_clients]))
    dataset_server_size = int(client_dataset_test_size*0.3)
    
    server_test_x = x_test[ client_dataset_test_size : int(client_dataset_test_size+dataset_server_size) ]
    server_test_y = y_test[ client_dataset_test_size : int(client_dataset_test_size+dataset_server_size) ]

    server_dataset = tf.data.Dataset.from_tensor_slices((server_test_x , server_test_y )).batch(batch_size)

    x_test = x_test[:client_dataset_test_size]
    y_test = y_test[:client_dataset_test_size]





    if (IsIID == True):
        
        print(colored('---------- IID = True ----------', 'green'))  
        
        # get train dataset   for client
        train_data_list = []
        start_train = 0
        for size in num_expamples_list_in_clients:
            client_i_train_dataset = list( zip( x_train, y_train ))[start_train : size+start_train]  
            train_data_list.append( preprocess_client_data( client_i_train_dataset )  )
            start_train  += size
        
        # get  test dataset for client
        test_data_list = []
        start_test = 0
        for test_size in [x*tt_rate for x in num_expamples_list_in_clients]:
            client_i_test_dataset =  list(zip( x_test , y_test ))[start_test:int(test_size)+start_test] 
            test_data_list.append(  preprocess_client_data( client_i_test_dataset )  )
            start_test += size
        
        #for test server

        return train_data_list, test_data_list , server_dataset
        
        
            
    else:
        ''' creates x non_IID clients'''

        
        
        print(colored('---------- IID = False ----------', 'green'))
        #create unique label list and shuffle
        
        unique_labels = np.unique(np.array(y_train))
        # random.shuffle(unique_labels)
        unique_labels = sorted(unique_labels)
        
        train_class = [None]*num_clients

        # classifar examples by unique label
        for (item , num_examples_client)  in zip(unique_labels , num_expamples_list_in_clients ):
            
            train_class[item] = [(image, label) for (image, label) in zip(x_train, y_train) if label == item][:num_examples_client]
        
        clients_dataset_list = []
        for dataset in train_class:
            clients_dataset_list.append( preprocess_client_data(dataset) )

        # dataset for server test


        return clients_dataset_list , server_dataset
    
    
# train_list , test_list  , server_data = generate_clients_data(num_expamples_list_in_clients=[1000] , IsIID=True)

# model = mnist_model(False)

# model.compile(optimizer= tf.keras.optimizers.SGD(0.01) , loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True) , metrics=['accuracy'])
# model.fit( train_list[0] , epochs=5  ,validation_data=test_list[0] , validation_freq=5 )
# loss , acc=model.evaluate(server_data)
# print('loss:{} , acc:{}'.format(loss,acc))
# train_list = tf.data.Dataset.from_tensors(train_list[0])

# print(tf.sets.size(train_list) , '\n')

# print(tf.sets.size(train_list[0]) , '\n')

# print(tf.sets.size(train_list[0][0]) , '\n')

   
