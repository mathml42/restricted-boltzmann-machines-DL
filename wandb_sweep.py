import wandb
from src.train import train
import argparse

def sweeper():
    parser = argparse.ArgumentParser(description="Initialize sweeps")

    parser.add_argument('-c', '--count', type=int, default=1, help="no of sweep count")
    args = parser.parse_args()

    sweep_config={'method':'bayes',
        'metric':{'name':'accuracy',
            'goal':'maximize'}}

    parameters_dict={    
        'rbm_epoch':{'values':[1]},
        'classifier_epoch':{'values':[3]},
        'k_steps_markov':{'values':[100, 200, 300]},
        'r_sample':{'values':[10, 20, 30]},
        'n_hidden':{'values':[64, 128, 256]},
        'learning_rate':{'values':[0.001]},
        'method':{'values':['cont_div']},
        'wandb_log':{'values':[True]}
        }

    proj_name="RBMs_experimenting"
    sweep_config['parameters']=parameters_dict
    sweep_id = wandb.sweep(sweep_config, project = proj_name)
    def train_with_name():
        wandb.init()
        config = wandb.config
        name = f"re{config.rbm_epoch}_ce{config.classifier_epoch}_k{config.k_steps_markov}_r{config.r_sample}_hd{config.n_hidden}_lr{config.learning_rate}_m{config.method}"
        wandb.run.name = name
        wandb.run.save
        train(**wandb.config)
    wandb.agent(sweep_id, train_with_name, project = proj_name, count = args.count)

if __name__ == "__main__":
    sweeper()