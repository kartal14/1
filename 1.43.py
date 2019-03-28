import numpy as np
import random

def Sigmoid(x):
    '''
    definition of sigmoid function and it's derivation
    '''
    from math import exp
    return 1.0 / (1.0 + exp(-x))
def SigmoidDerivate(y):
    return y * (1 - y)

def Tanh(x):
    '''
    definition of sigmoid function and it's derivation
    '''
    from math import tanh
    return tanh(x)
def TanhDerivate(y):
    return 1 - y*y


class BP_network:

    def __init__(self):
        #神经元个数
        self.d=0
        self.q=0
        self.l=0
        #神经元输出
        self.alpha_i=[]
        self.beta_h=[]
        self.y_j=[]
        #神经元权值
        self.v_ih=[]
        self.w_hj=[]
        #神经元阈值
        self.gamma_h=[]
        self.theta_j=[]
        #学习率
        self.eta1=[]
        self.eta2=[]
        #激活函数
        self.fun = {
            'Sigmoid': Sigmoid,
            'SigmoidDerivate': SigmoidDerivate,
            'Tanh': Tanh,
            'TanhDerivate': TanhDerivate,

            # for more, add here
        }

    def Creata_NN(self,input_num,hide_num,output_num,act_fun,learning_rate):
        self.d=input_num
        self.q=hide_num
        self.l=output_num

        self.i_v=np.zeros(self.d)
        self.h_v=np.zeros(self.q)
        self.o_v=np.zeros(self.l)

        self.v_ih=np.zeros([self.d,self.q])
        self.w_hj=np.zeros([self.q,self.l])
        for i in range(self.d):
            for h in range(self.q):
                self.v_ih[i][h]=random.random()
        for h in range(self.q):
            for j in range(self.l):
                self.w_hj[h][j]=random.random()
        print(self.v_ih)
        print(self.w_hj)
        self.gamma_h = np.zeros(self.q)
        self.theta_j = np.zeros(self.l)
        for h in range(self.q):
            self.gamma_h[h]=random.random()
        for j in range(self.l):
            self.theta_j[j]=random.random()
        print(self.gamma_h)
        print(self.theta_j)

        self.af=self.fun[act_fun]
        self.afd=self.fun[act_fun+'Derivate']

        self.eta1=np.ones(self.d)*learning_rate
        self.eta2=np.ones(self.q)*learning_rate
        print(self.eta1)
        print(self.eta2)


    def Predict(self,x):

        for i in range(self.d):
            self.i_v[i]=x[i]

        for h in range(self.q):
            total=0.0
            for i in range(self.d):
                total+=self.i_v[i]*self.v_ih[i][h]
            self.h_v[h]=self.af(total-self.gamma_h[h])

        for j in range(self.l):
            total=0.0
            for h in range(self.q):
                total+=self.h_v[h]*self.w_hj[h][j]
            self.o_v[j]=self.af(total-self.theta_j[j])




BPN1=BP_network()
BPN1.Creata_NN(2,3,1,act_fun='Sigmoid',learning_rate=0.2)

