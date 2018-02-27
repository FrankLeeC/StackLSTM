# -*- coding: utf-8 -*-

import numpy as np
import logging
import win_unicode_console
win_unicode_console.enable()

 
logger = logging.getLogger("stack_lstm")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
file_handler = logging.FileHandler("train.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
logger.removeHandler(file_handler)

def num2one_hot(n, size):
    targets = np.array([n]).reshape(-1)
    d = np.eye(size)[targets]
    return d.T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def dtanh(x):
    return 1 - x ** 2


def softmax(a):
    m = np.max(a)
    s = np.sum(np.exp(a - m))
    return np.exp(a - m) / s


class Param:

    def __init__(self, hsize, isize, osize):
        """
        isize: hsize+vsize(input layer) or hsize+hsize(hidden layer and output layer)
        osize: hsize(input layer and hidden layer) or vsize(outputlayer)
        """
        self.hsize = hsize
        self.isize = isize
        self.osize = osize
        self.wf = 0.5*np.random.rand(hsize, isize)
        self.bf = 0.5*np.random.rand(hsize, 1)
        self.wc = 0.5*np.random.rand(hsize, isize)
        self.bc = 0.5*np.random.rand(hsize, 1)
        self.wi = 0.5*np.random.rand(hsize, isize)
        self.bi = 0.5*np.random.rand(hsize, 1)
        self.wo = 0.5*np.random.rand(hsize, isize)
        self.bo = 0.5*np.random.rand(hsize, 1)
        self.wv = 0.5*np.random.rand(osize, hsize)
        self.bv = 0.5*np.random.rand(osize, 1)
    
    def save(self, index):
        np.save('./save/'+str(index)+'_wf', self.wf)
        np.save('./save/'+str(index)+'_bf', self.bf)
        np.save('./save/'+str(index)+'_wc', self.wc)
        np.save('./save/'+str(index)+'_bc', self.bc)
        np.save('./save/'+str(index)+'_wi', self.wi)
        np.save('./save/'+str(index)+'_bi', self.bi)
        np.save('./save/'+str(index)+'_wo', self.wo)
        np.save('./save/'+str(index)+'_bo', self.bo)
        np.save('./save/'+str(index)+'_wv', self.wv)
        np.save('./save/'+str(index)+'_bv', self.bv)


class Layer:

    def __init__(self, hsize, isize, osize, is_output=False):
        self.p = Param(hsize, isize, osize)
        self.lstm = LSTM(self.p, is_output)
        self.is_output = is_output
        self.err_arr = []
        self.batch = 0
        self.print_count = 100

    def sample_step(self, input_vector, h_prev, c_prev):
        return self.lstm.sample_step(input_vector, h_prev, c_prev)

    def forward(self, x_arr, y_arr=None):
        self.err_arr = []
        self.batch += 1
        out_arr = []
        h_prev = np.zeros((self.lstm.param.hsize, 1))
        c_prev = np.zeros((self.lstm.param.hsize, 1))
        loss = 0.0
        for i in range(x_arr.__len__()):
            x = x_arr[i]
            y_hat, h_prev, c_prev = self.lstm.step(x, h_prev, c_prev)
            out_arr.append(y_hat)
            if self.is_output:
                y = y_arr[i]
                loss += -np.dot(np.log(y_hat).T, y)
                self.err_arr.append((y_hat - y))
        if self.batch % self.print_count == 0 and self.is_output:
            logging.info(str(self.batch) + ' ---> loss: ' + str(loss))
            # print(self.batch, ' --> loss:', loss)
        return out_arr

    def backward(self, err=None):
        clip = 0.5
        dx_arr = None
        if self.is_output:
            dx_arr = self.lstm.backward(self.err_arr, clip)
        else:
            dx_arr = self.lstm.backward(err, clip)
        return dx_arr


class LSTM:

    def __init__(self, p, is_output=False):
        self.param = p
        self.is_output = is_output
        # self.is_input = is_input
        self.z_arr, self.f_arr, self.cs_arr, self.i_arr, self.c_arr, self.o_arr, self.h_arr, self.v_arr, self.y_arr = [], [], [], [], [], [], [], [], []
        self.init_adagrad()

    def init_adagrad(self):
        self.wf_g = Adagrad(np.zeros_like(self.param.wf))
        self.bf_g = Adagrad(np.zeros_like(self.param.bf))
        self.wc_g = Adagrad(np.zeros_like(self.param.wc))
        self.bc_g = Adagrad(np.zeros_like(self.param.bc))
        self.wi_g = Adagrad(np.zeros_like(self.param.wi))
        self.bi_g = Adagrad(np.zeros_like(self.param.bi))
        self.wo_g = Adagrad(np.zeros_like(self.param.wo))
        self.bo_g = Adagrad(np.zeros_like(self.param.bo))
        if self.is_output:
            self.wv_g = Adagrad(np.zeros_like(self.param.wv))
            self.bv_g = Adagrad(np.zeros_like(self.param.bv))

    def sample_step(self, x, h_prev, c_prev):
        z = np.row_stack((x, h_prev))
        f = sigmoid(np.dot(self.param.wf, z) + self.param.bf)  # hsize x 1
        cs = np.tanh(np.dot(self.param.wc, z) + self.param.bc)  # hsize x 1
        i = sigmoid(np.dot(self.param.wi, z) + self.param.bi)  # hsize x 1
        c = f*c_prev + i*cs  # hsize x 1
        o = sigmoid(np.dot(self.param.wo, z) + self.param.bo)  # hsize x 1
        h = np.tanh(c)*o  # hsize x 1
        y = h
        if self.is_output:
            v = np.dot(self.param.wv, h) + self.param.bv
            y = softmax(v)  # vsize x 1
        return y, h, c

    def clear(self):
        self.z_arr, self.f_arr, self.cs_arr, self.i_arr, self.c_arr, self.o_arr, self.h_arr, self.v_arr, self.y_arr = [], [], [], [], [], [], [], [], []     

    def step(self, x, h_prev, c_prev):   
        z = np.row_stack((x, h_prev))
        f = sigmoid(np.dot(self.param.wf, z) + self.param.bf)  # hsize x 1
        cs = np.tanh(np.dot(self.param.wc, z) + self.param.bc)  # hsize x 1
        i = sigmoid(np.dot(self.param.wi, z) + self.param.bi)  # hsize x 1
        c = f*c_prev + i*cs  # hsize x 1
        o = sigmoid(np.dot(self.param.wo, z) + self.param.bo)  # hsize x 1
        h = np.tanh(c)*o  # hsize x 1
        self.z_arr.append(z)
        self.f_arr.append(f)
        self.cs_arr.append(cs)
        self.i_arr.append(i)
        self.c_arr.append(c)
        self.o_arr.append(o)
        self.h_arr.append(h)
        y = h
        if self.is_output:
            v = np.dot(self.param.wv, h) + self.param.bv
            self.v_arr.append(v)
            y = softmax(v)  # vsize x 1
        self.y_arr.append(y)
        return y, h, c

    def backward(self, err, clip = 1.0):
        l = self.y_arr.__len__()
        dwf = np.zeros_like(self.param.wf)
        dbf = np.zeros_like(self.param.bf)
        dwc = np.zeros_like(self.param.wc)
        dbc = np.zeros_like(self.param.bc)
        dwi = np.zeros_like(self.param.wi)
        dbi = np.zeros_like(self.param.bi)
        dwo = np.zeros_like(self.param.wo)
        dbo = np.zeros_like(self.param.bo)
        dwv = np.zeros_like(self.param.wv)
        dbv = np.zeros_like(self.param.bv)
        dc_next = np.zeros((self.param.hsize, 1))
        dh_next = np.zeros((self.param.hsize, 1))
        dx_arr = [np.zeros((self.param.isize - self.param.hsize, 1))] * err.__len__()
        for i in range(l):
            j = l - i - 1
            dh = None
            if self.is_output:                
                dv = err[j]  # vsize x 1
                dwv += np.dot(dv, self.h_arr[j].T)  # vsize x hsize
                dbv += dv
                dh = np.dot(self.param.wv.T, dv) + dh_next # hsize x 1 
            else:
                dh = err[j] + dh_next
            do = dh*np.tanh(self.c_arr[j])  # hsize x 1
            dc = dh*self.o_arr[j]*dtanh(np.tanh(self.c_arr[j])) + dc_next  # hsize x 1
            dc_next = dc*self.f_arr[j]
            dwo +=  np.dot(do*dsigmoid(self.o_arr[j]), self.z_arr[j].T)  # hsize x vsize
            dbo += do*dsigmoid(self.o_arr[j])  # hsize x 1
            df = None
            if j == 0:
                df = dc*np.zeros((self.param.hsize, 1)) 
            else:
                df = dc*self.c_arr[j-1]
            di = dc*self.cs_arr[j]  # hsize x 1
            dcs = dc*self.i_arr[j]
            dwi += np.dot(di*dsigmoid(self.i_arr[j]), self.z_arr[j].T)  # hsize x vsize
            dbi += di*dsigmoid(self.i_arr[j])
            dwc += np.dot(dcs*dtanh(self.cs_arr[j]), self.z_arr[j].T)
            dbc += dcs*dtanh(self.cs_arr[j])
            dwf += np.dot(df*dsigmoid(self.f_arr[j]), self.z_arr[j].T)
            dbf += df*dsigmoid(self.f_arr[j])
            dz = np.dot(self.param.wf.T, df*dsigmoid(self.f_arr[j])) + np.dot(self.param.wc.T, dcs*dtanh(self.cs_arr[j])) + \
                np.dot(self.param.wi.T, di*dsigmoid(self.i_arr[j])) + np.dot(self.param.wo.T, do*dsigmoid(self.o_arr[j]))
            dh_next = dz[self.param.isize - self.param.hsize:]
            dx_arr[j] = dz[:self.param.isize - self.param.hsize]

        # update
        dwf = np.clip(dwf, -clip, clip)
        dbf = np.clip(dbf, -clip, clip)
        dwc = np.clip(dwc, -clip, clip)
        dbc = np.clip(dbc, -clip, clip)
        dwi = np.clip(dwi, -clip, clip)
        dbi = np.clip(dbi, -clip, clip)
        dwo = np.clip(dwo, -clip, clip)
        dbo = np.clip(dbo, -clip, clip)
        if self.is_output:
            dwv = np.clip(dwv, -clip, clip)
            dbv = np.clip(dbv, -clip, clip)
        self.update(dwf, dbf, dwc, dbc, dwi, dbi, dwo, dbo, dwv, dbv)
        self.clear()
        return dx_arr

    def update(self, dwf, dbf, dwc, dbc, dwi, dbi, dwo, dbo, dwv, dbv):
        self.param.wf += self.wf_g.get_grad(dwf)
        self.param.bf += self.bf_g.get_grad(dbf)
        self.param.wc += self.wc_g.get_grad(dwc)
        self.param.bc += self.bc_g.get_grad(dbc)
        self.param.wi += self.wi_g.get_grad(dwi)
        self.param.bi += self.bi_g.get_grad(dbi)
        self.param.wo += self.wo_g.get_grad(dwo)
        self.param.bo += self.bo_g.get_grad(dbo)
        if self.is_output:
            self.param.wv += self.wv_g.get_grad(dwv)
            self.param.bv += self.bv_g.get_grad(dbv)


class Adagrad:

    def __init__(self, init_n, learning_rate=0.08, epsilon=0.00001):
        self.prev_n = init_n
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def get_grad(self, grad):
        n = self.prev_n + grad*grad
        self.prev_n = n
        g = -self.learning_rate*grad/(np.sqrt(n+self.epsilon))
        return g


class Deep_RNN:

    def __init__(self, layer_size=3):
        self.layer_size = layer_size
        self.word2int = {}
        self.int2word = {}
        self.layers = []

    def build_layer(self):
        self.hsize = 500
        for i in range(self.layer_size):
            osize = self.hsize
            isize = self.hsize + self.hsize
            is_output = False
            if i == self.layer_size - 1:
                osize = self.vsize
                is_output = True
            if i == 0:
                isize = self.hsize + self.vsize
            self.layers.append(Layer(self.hsize, isize, osize, is_output))

    def preprocess(self, path):
        self.data = open(path, mode='r', encoding='utf-8').read()
        self.length = self.data.__len__()        
        self.sd = list(set(self.data))
        self.vsize = self.sd.__len__()
        print('length:', self.length, '  vsize:', self.vsize)
        for i, v in enumerate(self.sd):
            self.word2int[v] = i
            self.int2word[i] = v

    def sample(self, length=20):
        n = np.random.randint(0, self.vsize)
        input_vector = num2one_hot(n, self.vsize)
        word = []
        word.append(input_vector)
        h_prev_arr, c_prev_arr = [], []
        for i in range(self.layer_size):
            h_prev_arr.append(np.zeros((self.hsize, 1)))
            c_prev_arr.append(np.zeros((self.hsize, 1)))
        for i in range(length):
            for j in range(self.layer_size):
                input_vector, h_prev_arr[j], c_prev_arr[j] = self.layers[j].sample_step(input_vector, h_prev_arr[j], c_prev_arr[j])
            word.append(input_vector)
        rs = []
        for i in word:
            m = np.argmax(i)      
            rs.append(self.int2word[m])
        return ''.join([x for x in rs])

    def save(self):
        f = open('./save/layer_size', 'w', encoding='utf-8')
        f.write(str(self.layer_size))
        f.flush()
        f.close()
        for i, v in enumerate(self.layers):
            v.lstm.param.save(i)
        f = open('./save/word', 'w', encoding='utf-8')
        for k in self.word2int.keys():
            f.write(k+':'+str(self.word2int[k])+'\n')
        f.flush()
        f.close()

    def train(self):
        start = 0
        num_step = 25        
        while True:
            if start >= self.length:
                break
            w = self.data[start:start+num_step]
            start += num_step
            x_arr = []
            y_arr = []
            for j, v in enumerate(w):
                if j < w.__len__() - 1:
                    x_arr.append(num2one_hot(self.word2int[v], self.vsize))
                    y_arr.append(num2one_hot(self.word2int[w[j+1]], self.vsize))
            for k, v in enumerate(self.layers):
                if k < self.layer_size - 1:
                    x_arr = v.forward(x_arr)
                else:
                    output = v.forward(x_arr, y_arr)
            err = None
            for k in range(self.layer_size):
                idx = self.layer_size - k - 1
                if k == 0:
                    err = self.layers[idx].backward()
                else:
                    err = self.layers[idx].backward(err)
                    

if __name__ == '__main__':
    rnn = Deep_RNN(2)
    rnn.preprocess('./five_poem.txt') 
    rnn.build_layer()   
    sample_count = 1
    try:
        for i in range(1000):
            rnn.train()
            if i % sample_count == 0:
                word = rnn.sample()
                logging.info(str(i) + ' ---> sample: ' + word)
                # print(i, ' --> sample: ', word)
    except KeyboardInterrupt as e:
        logging.error('stop!')
        # print('stop!')
    finally:
        logging.error('over!')
        # print('over!')
        rnn.save()
        word = rnn.sample()
        logging.info(word)
        # print(word)