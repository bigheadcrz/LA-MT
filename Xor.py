# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

hidden_layer=2
input_layer=2
out_layer=1
train_data1=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_data2=np.array([[0], [1], [1], [0]])

def init_weights(Layer_in,Layer_out,E): #初始化权重矩阵
    e=E
    W=np.random.rand(Layer_out,Layer_in+1)*2*e-e #数值范围0~1
    return W

def sigmoid(x): #sigmoid
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_gradient(z):#sigmoid求偏导
    g = np.multiply(sigmoid(z), (1 - sigmoid(z)))
    return g

def int_to_bin(num):#十进制转二进制
    a=int(num)
    b=bin(a)
    c=''
    if a<=1:
        c='00'+b[2]
    elif 1<a and a<=3:
        c='0'+b[2]+b[3]
    else:
        c=b[2]+b[3]+b[4]
    return c

def cost_func(w1,w2,t1,t2):
    m=t1.shape[0]
    #计算所有参数的偏导数
    D1=np.zeros(w1.shape) #▲1
    D2=np.zeros(w2.shape) #▲2
    h_pro=np.zeros((m,1)) #样本预测值
    for t in range(m):
        a_1 = np.vstack((np.array([[1]]), t1[t:t + 1, :].T))   #3*1
        z_2 = np.dot(w1, a_1)   # 2*1
        a_2 = np.vstack((np.array([[1]]), sigmoid(z_2)))    # 3*1
        z_3 = np.dot(w2, a_2)   # 1*1
        a_3 = sigmoid(z_3)
        h = a_3  # 预测值h=a_3, 1*1
        h_pro[t, 0] = h
        #后向传播计算梯度
        dt_3=h-t2[t:t + 1, :].T # 最后一层每一个单元的误差,δ_3,1*1
        dt_2=np.multiply(np.dot(w2[:, 1:].T,dt_3), sigmoid_gradient(z_2)) # 第二层每一个单元的误差（不包括偏置单元）,δ_2,2*1
        D2=D2+np.dot(dt_3,a_2.T) # 第二层所有参数的误差, 1*3，权重矩阵的偏导数
        D1=D1+np.dot(dt_2,a_1.T) # 第一层所有参数的误差, 2*3，权重矩阵的偏导数
    w1_grad=(1.0/m)*D1
    w2_grad=(1.0/m)*D2
    J = (1.0 / m) * np.sum(-t2 * np.log(h_pro) - (np.array([[1]]) - t2) * np.log(1 - h_pro))#代价函数
    return {'w1_grad':w1_grad,'w2_grad':w2_grad,'J':J,'h':h_pro}

#随机初始化权重矩阵
w1=init_weights(input_layer,hidden_layer,E=1)
w2=init_weights(hidden_layer,out_layer,E=1)

iter_times=10000 #迭代次数
learning_rates=0.5 #学习率
result={'J': [], 'h': []}
theta_s={}

for i in range(iter_times):
    cost_func_result=cost_func(w1=w1,w2=w2,t1=train_data1,t2=train_data2)
    w1_g=cost_func_result.get('w1_grad')
    w2_g = cost_func_result.get('w2_grad')
    J=cost_func_result.get('J')
    h_current=cost_func_result.get('h')
    w1 -= learning_rates*w1_g
    w2 -= learning_rates*w2_g
    result['J'].append(J)
    result['h'].append(h_current)
    # print(i, J, h_current)
    if i == 0 or i == (iter_times - 1):
        if (i == 0):
            print("随机初始化得到的参数")
        else:
            print("训练后得到的参数")
        print('w1', w1)
        print('w2', w2)
        theta_s['w1_' + str(i)] = w1.copy()
        theta_s['w2_' + str(i)] = w2.copy()

plt.plot(result.get('J'))
plt.show()

print("第一次迭代和最后一次迭代得到的参数")
for item in theta_s.items():
    print(item)

print("初始参数预测出来的值")
print(result.get('h')[0])
print("最后一次得到的参数预测出来的值")
print(result.get('h')[-1])
if (result['h'][-1][0][0])<0.01 and (result['h'][-1][3][0])<0.01 and (1-(result['h'][-1][1][0])) <0.01 and (1-(result['h'][-1][2][0]))<0.01:
    print("收敛成功")
    while True:
        print("请输入X：", end='')
        x = input()
        print("请输入Y：", end='')
        y = input()
        if int(x) < 0 or int(x) > 7 or int(y) < 0 or int(y) > 7:
            print("输入有误")
            break
        X = int_to_bin(x)
        Y = int_to_bin(y)
        C = ''
        for i in range(0, 3):
            if X[i] == '0' and Y[i] == '0':
                C += str(int(round(result['h'][-1][0][0])))
            elif X[i] == '0' and Y[i] == '1':
                C += str(int(round(result['h'][-1][1][0])))
            elif X[i] == '1' and Y[i] == '0':
                C += str(int(round(result['h'][-1][2][0])))
            elif X[i] == '1' and Y[i] == '1':
                C += str(int(round(result['h'][-1][3][0])))
        print("异或结果：", end='')
        print(int(C, 2))
else:
    print("收敛失败")
