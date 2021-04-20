import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import nest_asyncio
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
nest_asyncio.apply()

'''
自定义联邦数据类型
'''
#在联邦学习中，采样的本地设备中数据的类型对于外界是不可知的
#如果所有本地设备中的数据类型都一样，则被称为第一好的数据
#联邦数据类型有两种定义方式，假设本地设备中的所有数据都一样

#定义所有设备上的数据类型都为32位浮点型
#如果联邦数据只有一种类型，则不管该数据是不是联邦数据类型都可以认为是完全等价的
federated_float_on_clients = tff.type_at_clients(tf.float32 )

#完整显示联邦数据类型
# print( str(federated_float_on_clients) , '\n')
#第一个参数表示联邦数据成员类型 ， 第二个参数表示联邦数据部署位置
# print( str(federated_float_on_clients.member),'\n' , str(federated_float_on_clients.check_placement))

#联邦数据有两种定义方式，第一种已知分布式设备上的所有数据都相同
#第二种分布式设备上数据不全相同（默认）federated_float_on_clients.all_equal = False
# print(str(federated_float_on_clients.all_equal))

#创建本地设备不全相同的联邦数据表示类型
#以y = ax+b为例
differ_simple_regression_model_type = (
    #类型不一定要是数值类型
    tff.StructType([('a' , tf.float32) , ('b' , tf.string) ,('c' , tf.float64)])
)
# print(str(differ_simple_regression_model_type))
#当联邦数据类型全都一致时，可以表示为
#该联邦数据类型具有对称性，因此通常使用术语xyz联邦来指成员组成类似于xyz的联邦值
same_simple_regression_model_type = (
    tff.StructType([('a' , tf.float32) , ('b' , tf.float32)])
)
# print(str(tff.type_at_clients(differ_simple_regression_model_type , all_equal=True)))


'''
联邦计算（关键单元）
'''
#TFF是一种强类型的函数式编程
#可以使用联邦数据作为输入，同时使用联邦数据作为输出
#示例：读取所有温度传感器进行计算
#温度传感器就是客户端，将传感器读取的温度构造成联邦数据进行输入

#指定联邦计算输入类型
#与tf.function不同，tff.federated_computation既不是tf也不是python
#而是一种内部独立于平台的粘合语言的分布式系统规范
#在使用前明确使用输入和输出的参数类型

#装饰器
#普通python函数计算，
#使用该装饰器即可封装成tff类型，可直接使用python数据作为输入

#无论是tff代码封装python函数还是在python函数中使用tff代码，
# 其都是平台无关的 ， 使用装饰器时，在定义时，其中的python代码就在执行

#tf代码必须在tff.tf_computation装饰器装饰的代码中使用
@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
    #计算并输出平均联邦数据
    #联邦计算装饰器，
    # 装饰的python函数体中的函数调用必须使用的时tFF类型函数
    return tff.federated_mean(sensor_readings)

#tff计算通常被建模为函数
#可以通过类型签名来查看函数的类型签名
# print(get_average_temperature.type_signature,'\n')

#联邦计算，为了便于调试可以使用同型或非同型的数据，
# 在本例中可以直接使用python中的list类型
# print(get_average_temperature([68.5, 70.3 , 69.8]))

'''
tf逻辑，声明tf计算：
    在大多数情况下，编写的tff代码都和tf代码一样。
    只不过需要装饰器装饰tff.tf_computation
'''
#示例
@tff.tf_computation(tf.float32)
def add_half(x):
    #tf代码封装在tff.tf_computation装饰器中
    return tf.add(x , 0.5)

#tff.tf_computation装饰器装饰过的tf代码就如同tff代码一样使用
#tff.tf_computation封装的tf代码没有部署位置的属性，无法处理或输出联邦数据
# print(add_half.type_signature)

#可以将一个函数作为参数被另一个函数调用
@tff.federated_computation(tff.type_at_clients(tf.float32))
def add_half_on_clients(x):
    #x可以是一个列表，以列表中的每个元素作为add_half参数，逐点相加
    return tff.federated_map(add_half , x)

# print(add_half_on_clients.type_signature,'\n')
# print(add_half_on_clients([1.0,2.0,3.0,4.0,5.0]))

'''
python代码使用tff.federatde_computation装饰，tf代码使用tff.tf_computation装饰
区别：
    前者不能直接在python代码中使用tf代码
    后者tf代码被序列化为tf图

    tff.tf_computation临时禁用了立即执行，以便捕获计算结构
'''
#错误示范
try:
    #constant-10不在tff.tf_computation装饰器中，无法添加到tf图中
    # constant_10 = tf.constant(10.)
    @tff.tf_computation(tf.float32)
    def add_ten(x):
        return x+ tf.constant(10.)
except Exception as err:
    print(err)

# print(add_ten(1))

#正确示范
def get_constant(x):
    return x
@tff.tf_computation(tf.float32 , tf.float32)
def add_constant(x , constant):
    return x + get_constant(constant)
# print(type(add_constant(1.0,2.0))) 

#使用tf.data.Datasets
#tff.tf_computation装饰器能够像平常使用tf数据集一样使用
#在tff中，数据集需要声明成tff.SequenceType类型，也就是tff中的一种序列
#该序列能够容纳复杂的元素
float32_sequence = tff.SequenceType(tf.float32)

#计算本地设备的平均温度操作使用Dataset.reduce
@tff.tf_computation(tff.SequenceType(tf.float32))
def get_local_temperature_average(local_temperature):
    sum_and_count = (
        local_temperature.reduce((0.0,0),lambda x,y: (x[0]+y , x[1]+1))
    )
    return sum_and_count[0] / tf.cast(sum_and_count[1] , tf.float32 )
#计算全局设备的平均温度操作
@tff.federated_computation(tff.type_at_clients(tff.SequenceType(tf.float32)))
def get_global_temperature_average(sersor_readings ):
    #federated_mean接收权重作为第二个可选参数
    return tff.federated_mean(
        tff.federated_map(get_local_temperature_average , sersor_readings)
        # weight=value_weight
    )

# print(get_global_temperature_average([[68.0, 70.0], [71.0], [68.0, 72.0, 70.0]]))

#通过使用tff.SequenceType构造器，可以支持嵌入式结构
@tff.tf_computation(tff.SequenceType(collections.OrderedDict([('A' , tf.float32) , ('B' , tf.float32)])))
def fun(ds):
    print('element_structure = {}'.format(ds.element_spec))
    return ds.reduce(np.float32(0) , lambda total , x :total+x['A']*x['B'])

print(fun([{'A':2 , 'B':3} , {'A':4 , 'B':5}]))

