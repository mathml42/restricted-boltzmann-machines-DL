import numpy as np
import wandb
import time
class RBM():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, n_hidden, k_steps_markov, 
                 r_sample, rbm_epoch, classifier_epoch, learning_rate, method, wandb_log=False):
        self.n_visible = x_train.shape[1]
        self.n_hidden = n_hidden
        self.k_steps_markov = k_steps_markov
        self.r_sample = r_sample
        self.n_class = y_train.shape[1]
        self.x_train_len = x_train.shape[0]
        self.x_val_len = x_val.shape[0]
        self.x_train = x_train
        self.x_val = x_val
        self.y_val = y_val
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_test_len = x_test.shape[0]
        self.learning_rate = learning_rate
        self.rbm_epoch = rbm_epoch
        self.classifier_epoch = classifier_epoch
        self.wandb_log = wandb_log
        self.method = method

    # parameter initialization
    def rbm_params_init(self):
        rbm_params = {}
        rbm_params['W'] = np.random.randn(self.n_hidden, self.n_visible)*np.sqrt(6./(self.n_visible + self.n_hidden))
        rbm_params['h_bias'] = np.zeros((self.n_hidden, 1), dtype=np.float32)
        rbm_params['v_bias'] = np.zeros((self.n_visible, 1), dtype=np.float32)
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
    
    def rbm_train(self, parameters):
        W = parameters['W']
        h_bias = parameters['h_bias']
        v_bias = parameters['v_bias']
        if self.method == 'gibbs':
            for i in range(self.x_train_len):

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
                        v_given_h = self.sigmoid(np.dot(np.transpose(W),h_sample) + v_bias)
                        v_sample = np.random.binomial(1, v_given_h)

                        dw = dw + 1/self.n_hidden * (np.dot(self.sigmoid(np.dot(W, v_sample)+h_bias),np.transpose(v_sample)))
                        dv = dv + 1/self.n_hidden * (v_sample)
                        dh = dh + 1/self.n_hidden * (self.sigmoid(np.dot(W, v_sample) + h_bias))
                
                W = W + self.learning_rate*(np.dot(self.sigmoid(np.dot(W,v_init)+h_bias), np.transpose(v_init)) - dw)
                v_bias = v_bias + self.learning_rate * (v_init - dv)
                h_bias = h_bias + self.learning_rate * (self.sigmoid(np.dot(W, v_init) + h_bias) - dh)
                
                if i % 10 == 0:
                    print(i)

            parameters['W'] = W
            parameters['h_bias'] = h_bias
            parameters['v_bias'] = v_bias
            print("Training Complete")

            return parameters
        
        elif self.method == "cont_div":
            for i in range(self.x_train_len):
                v_sample = self.x_train[i]
                v_init = self.x_train[i]

                for t in range(self.k_steps_markov):
                    h_given_v = self.sigmoid(np.dot(W, v_sample) + h_bias)
                    h_sample = np.random.binomial(1, h_given_v)
                    v_given_h = self.sigmoid(np.dot(np.transpose(W), h_sample) + v_bias)
                    v_sample = np.random.binomial(1, v_given_h)

                W += self.learning_rate * (np.dot(self.sigmoid(np.dot(W, v_init) + h_bias),np.transpose(v_init)) - np.dot(self.sigmoid(np.dot(W, v_sample) + h_bias),np.transpose(v_sample)))
                v_bias += self.learning_rate * (v_init-v_sample)
                h_bias = h_bias + self.learning_rate * (self.sigmoid(np.dot(W,v_init)+h_bias) - self.sigmoid(np.dot(W,v_sample)+h_bias))

                if i % 15000 == 0:
                    print(i)
            parameters["W"] = W
            parameters["h_bias"] = h_bias
            parameters["v_bias"] = v_bias
            print("Training Complete")

            return parameters
        else:
            NameError("Method not found.")


    
    def get_hidden_rep(self, x, parameters):
        W = parameters['W']
        h_bias = parameters['h_bias']
        hidden_prob = self.sigmoid(np.dot(W,x) + h_bias)
        hidden_rep = np.random.binomial(1, hidden_prob)
        return hidden_rep

    def classifier_train(self, classifier_params):
        W = classifier_params['W']
        b = classifier_params['b']

        for epoch in range(self.classifier_epoch):
            for i in range(self.x_val_len):
                hidden_rep = self.get_hidden_rep(self.x_val[i], self.rbm_params)
                pre_out = np.dot(W,hidden_rep) + b
                y_hat = self.softmax(pre_out)
        
                dW = np.dot(-(self.y_val[i]-y_hat),np.transpose(hidden_rep))
                db = -(self.y_val[i]-y_hat)
                # Update Classifier weights
                W = W - self.learning_rate*dW
                b = b - self.learning_rate*db

        classifier_params["W"] = W
        classifier_params["b"] = b 
        
        return classifier_params

    def rbm_classifier(self):
        classifier_params = self.classifier_params_init()
        self.rbm_params = self.rbm_params_init()
        # wandb.login()
        # wandb.init(project="RBMs_experimenting")

        for j in range(self.rbm_epoch):
            start_time = time.time()
            self.rbm_params = self.rbm_train(self.rbm_params)
            classifier_params = self.classifier_train(classifier_params)

            accuracy = 0.0
            loss = 0.0

            for i in range(self.x_test_len):
                h = self.get_hidden_rep(self.x_test[i], self.rbm_params)
                y_hat = self.softmax(np.dot(classifier_params['W'],h) + classifier_params['b'])

                if y_hat.argmax() == self.y_test[i].argmax():
                    accuracy += 1
                loss += (-1 * np.sum(np.multiply(self.y_test[i], np.log(y_hat + 1e-10))))
            accuracy = accuracy/self.x_test_len
            loss = loss / self.x_test_len
            elasped_time = time.time()- start_time
            if elasped_time >= 60:
                minutes = int(elasped_time // 60)
                seconds = int(elasped_time % 60)
                time_str = f"{minutes}:{seconds}"
            else:
                time_str = f"{elasped_time:.2f}s"
            print(f"Epoch : {j}, Accuracy : {accuracy}, Loss : {loss}, Time taken : {time_str}")
            if self.wandb_log == True:
                wandb.log({"Accuracy": accuracy, "Loss":loss, 'Epoch':j})
        if self.wandb_log == True:
            wandb.finish()    
            
        return self.rbm_params,classifier_params