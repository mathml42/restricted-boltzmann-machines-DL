from src.train import train
import argparse


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a Restricted Boltzmann Machine Classifier.')

    # Add arguments
    parser.add_argument('-re', '--rbm_epoch', type=int, default=3, help='Number of RBM training epochs')
    parser.add_argument('-ce', '--classifier_epoch', type=int, default=3, help='Number of classifier training epochs')
    parser.add_argument('-k', '--k_steps_markov', type=int, default=10, help='Number of Markov steps')
    parser.add_argument('-r', '--r_sample', type=int, default=5, help='Number of samples to draw')
    parser.add_argument('-hd', '--n_hidden', type=int, default=128, help='Number of hidden units')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('-m', '--method', type=str, default='cont_div', help='Sampling method to use')
    parser.add_argument('-w', '--wandb', type=bool, default=False, help='Whether to log training')
    # parser.add_argument('-e', '--exp_count', type=int, default=1, help='No of experiment to be log')

    # Parse the arguments
    args = parser.parse_args()

    # Call the train function with parsed arguments
    rbm_params, classifier_params = train(
        rbm_epoch=args.rbm_epoch,
        classifier_epoch=args.classifier_epoch,
        k_steps_markov=args.k_steps_markov,
        r_sample=args.r_sample,
        n_hidden=args.n_hidden,
        learning_rate=args.learning_rate,
        method=args.method,
        wandb_log=args.wandb,
        # train_count=args.exp_count
    )

if __name__ == "__main__":
    main()