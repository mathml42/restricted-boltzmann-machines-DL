import numpy as np

class RBM():
    def __init__(self, x_train,n_visible, n_hidden, k_steps_markov, r_sample, n_class):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k_steps_markov = k_steps_markov
        self.r_sample = r_sample
        self.n_class = n_class
        self.train_len = x_train.shape[0]

    # parameter initialization
    def rbm_params_init(self):
        rbm_params = {}
        rbm_params['W'] = np.random.randb(self.n_hidden, self.n_visible)*np.sqrt(6./(self.n_visible + self.n_hidden))
        rbm_params['b'] = np.zeros((self.n_hidden, 1), dtype=np.float32)
        rbm_params['c'] = np.zeros((self.n_hidden, 1), dtype=np.float32)
        return rbm_params
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def classifier_params_init(self):
        classifier_params = {}
        classifier_params['W'] = np.random.randn(self.n_class, self.n_hidden)*np.sqrt(6./(self.n_class + self.n_hidden))
        classifier_params['b'] = np.zeros((self.n_class, 1), dtype=np.float32)
        return classifier_params
    
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def rbm_train(self, parameters, learning_rate):
        W = parameters['W']
        h_bias = parameters['h_bias']
        v_bias = parameters['v_bias']

        for i in range(self.train_len):

            v_init = self.x_train[i]
            v_sample = np.random.randint(2, size=np.shape(self.x_train[i]))
            dw = np.zeros(np.shape(W))
            dv = np.zeros(np.shape(v_bias))
            dh = np.zeros(np.shape(h_bias))

            for t in range(self.n_visible + self.n_hidden):
                if t < self.n_visible:

                    h_given_v = self.sigmoid(np.dot(W, v_sample)+h_bias)
                    h_sample = np.random.binomial(1, h_given_v)
                    v_given_h = self.sigmoid(np.dot(np.transpose(W),h_sample) + v_bias)
                    v_sample = np.random.binomial(1, v_given_h)
                
                else:
                    h_given_v = self.sigmoid(np.dot(W, v_sample) + h_bias)
                    h_sample = np.random.binomial(1, h_given_v)
                    v_given_h = self.sigmoid(np.dot(np.transpose(W),v_sample) + v_bias)
                    v_sample = np.random.binomial(1, v_given_h)

                    