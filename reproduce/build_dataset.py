from argparse import ArgumentParser

from MoleculeBench import train_val_test_split, dataset_info

from deepchem.data import CSVLoader
from deepchem.feat import CircularFingerprint
from deepchem.trans import BalancingTransformer
from deepchem.utils.data_utils import save_dataset_to_disk


def save_dataset_splits(dataset, data_seed):
    info = dataset_info(dataset)
    loader = CSVLoader(tasks=info.task_columns,
                       featurizer=CircularFingerprint(size=1024),
                       feature_field=info.smiles_column)
    data_list = loader.create_dataset(inputs=info.filtered_path,
                                      data_dir=f'./reproduce/dataset/{dataset}',
                                      shard_size=8192)

    train_indices, val_indices, test_indices = \
        train_val_test_split(dataset=dataset, return_indices=True, random_state=data_seed)

    train = data_list.select(indices=train_indices)
    val = data_list.select(indices=val_indices)
    test = data_list.select(indices=test_indices)

    transformer = BalancingTransformer(train)
    train = transformer.transform(dataset=train)
    val = transformer.transform(dataset=val)
    test = transformer.transform(dataset=test)

    save_dir = f'./reproduce/dataset/{dataset}/data_seed={data_seed}' \
               if info.splitting == 'random' else \
               f'./reproduce/dataset/{dataset}/scaffold_split'

    save_dataset_to_disk(save_dir=save_dir,
                         train=train,
                         valid=val,
                         test=test,
                         transformers=[transformer])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace'], required=True)
    parser.add_argument('--data_seed', type=int, default=42)
    args = parser.parse_args()

    save_dataset_splits(args.dataset, args.data_seed)
