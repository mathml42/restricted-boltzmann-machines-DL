from data_loader import load_data
from RBMs import RBM
import wandb

rbm_epoch = 1,
classifier_epoch = 1,
k_steps_markov = 10,
r_sample = 5,
n_hidden = 80,
learning_rate = 0.001
method = "cont_div"
wandb_log = False

def train(n_hidden, k_steps_markov, r_sample,
          rbm_epoch, classifier_epoch, learning_rate, method, wandb_log):
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    RBM_classify = RBM(x_train, y_train, x_val, y_val,  x_test, y_test, n_hidden, k_steps_markov, 
                       r_sample, rbm_epoch, classifier_epoch, learning_rate, method, wandb_log)
    rbm_params, classifier_params = RBM_classify.rbm_classifier()

    return rbm_params, classifier_params

if __name__=="__main__":
    train()
