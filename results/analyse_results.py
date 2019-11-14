import os
import argparse
import pandas as pd


"""
    This script will navigate over all subfolders of args['path'],
    find the .tsv files with the classification of the respective
    models on validation dataset and will compute by default the
    harmonic mean on summary_of_results_{something}.tsv
"""

def compute_results(dataset, function):
    if function == 'arithmetic_mean':
        return dataset['Accuracy'].mean() 
    elif function == 'harmonic_mean':
        aux = [1. / x if x > .0 else 100 for x in dataset['Accuracy']]
        denominator = .0
        for x in aux: denominator += x
        return len(aux) / denominator
    else:
        print('Function to compute mean not recognized.')
        exit()


# Arguments
parser = argparse.ArgumentParser(
    description='Analyse the results obtained by the pruned NN.'
)
parser.add_argument('-p', '--path', help='Path to find the results in .tsv', required=True)
parser.add_argument('-c', '--mean_type', help='Way to compute the best pruned model', default='harmonic_mean')
args = vars(parser.parse_args())


dt = []
for (root, dirs, files) in os.walk(args['path']):
    if 'SEARCH' in root:
        folder = root.split('/')[-1]
        hyperparameters = folder.split('__')
        line = [x.split('-')[-1] for x in hyperparameters]
        line.append(root.split('/')[1])
        results = pd.read_csv(root + '/' + files[0], sep='\t')
        line.append(compute_results(results, args['mean_type']))
        dt.append(line)
dt = pd.DataFrame(dt, columns=['SEARCH', 'ALPHA', 'GAMMA', 'PRUNE_TYPE', 'PRUNE_PERCENT', 'EPS', 'REWARD_TYPE', 'Q_UPDATE', 'ACCURACY'])
name = args['path'].split('/')[1]
dt.to_csv('summary_of_results_{}.tsv'.format(name), sep='\t')