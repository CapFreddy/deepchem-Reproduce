import os
import pickle
from functools import partial
from argparse import ArgumentParser

import numpy as np

from MoleculeBench import dataset_info

from deepchem.metrics import Metric, roc_auc_score
from deepchem.molnet.preset_hyper_parameters import hps
from deepchem.utils.data_utils import load_dataset_from_disk
from deepchem.molnet.run_benchmark_models import benchmark_classification


def kernelsvm_multitask_model_builder(tasks, **hyper_parameters):
    from sklearn.svm import SVC
    from deepchem.models import SklearnModel
    from deepchem.models.multitask import SingletaskToMultitask

    C, gamma, model_dir = hyper_parameters['C'], hyper_parameters['gamma'], hyper_parameters['model_dir']

    def kernelsvm_sklearn_model_builder(model_dir):
        model = SVC(C=C, gamma=gamma, class_weight='balanced', probability=True)
        return SklearnModel(model, model_dir)

    return SingletaskToMultitask(tasks, kernelsvm_sklearn_model_builder, model_dir=model_dir)


def rf_multitask_model_builder(tasks, **hyper_parameters):
    from sklearn.ensemble import RandomForestClassifier
    from deepchem.models import SklearnModel
    from deepchem.models.multitask import SingletaskToMultitask

    n_estimators, model_dir = hyper_parameters['n_estimators'], hyper_parameters['model_dir']

    def rf_sklearn_model_builder(model_dir):
        model = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', n_jobs=-1)
        return SklearnModel(model, model_dir)

    return SingletaskToMultitask(tasks, rf_sklearn_model_builder, model_dir=model_dir)


def train(dataset, data_seed, run_seed, model, hyperparam_search):
    info = dataset_info(dataset)
    tasks = info.task_columns
    dataset_dir = f'./reproduce/dataset/{dataset}/data_seed={data_seed}' \
                  if info.splitting == 'random' else \
                  f'./reproduce/dataset/{dataset}/scaffold_split'
    result_dir = f'./reproduce/saved_models/{model}/{dataset}/data_seed={data_seed}/run_seed={run_seed}' \
                 if info.splitting == 'random' else \
                 f'./reproduce/saved_models/{model}/{dataset}/run_seed={run_seed}'
    os.makedirs(result_dir, exist_ok=True)

    _, all_datasets, transformers = load_dataset_from_disk(dataset_dir)
    train_dataset, valid_dataset, test_dataset = all_datasets

    if model == 'kernelsvm':
        model_builder = partial(kernelsvm_multitask_model_builder, tasks=tasks)
    elif model == 'rf':
        model_builder = partial(rf_multitask_model_builder, tasks=tasks)

    if hyperparam_search:
        from deepchem.hyper import GaussianProcessHyperparamOpt

        search_mode = GaussianProcessHyperparamOpt(model_builder)
        _, hyper_parameters, _ = search_mode.hyperparam_search(
            params_dict=hps[model],
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            metric=Metric(roc_auc_score, np.mean),
            output_transformers=transformers,
            logdir=result_dir,
        )

        with open(f'{result_dir}/hyperparams.pkl', 'wb') as fout:
            pickle.dump(hyper_parameters, fout)
    else:
        hyper_parameters = hps[model]

    all_scores = benchmark_classification(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        tasks=tasks,
        transformers=transformers,
        n_features=1024,
        metric=Metric(roc_auc_score, np.mean),
        model=model,
        test=True,
        hyper_parameters=hyper_parameters,
        seed=run_seed,
        model_dir=f'{result_dir}/saved_model'
    )

    print([score[model]['mean-roc_auc_score'] for score in all_scores])

    with open(f'{result_dir}/test_roc_auc_score.txt', 'w') as fout:
        fout.write(str(all_scores[2][model]['mean-roc_auc_score']))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace'], required=True)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--run_seed', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['kernelsvm', 'rf'], required=True)
    parser.add_argument('--hyperparam_search', action='store_true')
    args = parser.parse_args()

    train(args.dataset, args.data_seed, args.run_seed, args.model, args.hyperparam_search)
