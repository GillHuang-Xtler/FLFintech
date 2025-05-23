import argparse
from src.fl_base import FederatedLearningBase as FLClass
# from src.fl_two_variants import FLTwoVariants as FLClass
from src.gradient_inversion import GradientInversion

def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=True, action='store_true', help='using cuda')
    args.add_argument('--num_clients', type=int, default=10)
    args.add_argument('--save_path', type=str, default='res')
    args.add_argument('--stock_symbols', nargs='+', default=['ibm', 'aapl', 'abb', 'jpm', 'kep', 'naz', 'kro', 'oi', 'pbr', 'qtm'], type=str)

    args.add_argument('--global_rounds', type=int, default=100)

    args = args.parse_args()
    return args

def run_fl_training(args):
    fl = FLClass(args)
    fl.load_data()
    fl.set_nets()
    fl.federated_train()


def run_predict(dataset_idx, path='res/final_global_model.pth'):
    fl = FLClass(args)
    fl.load_data()
    fl.set_nets(path=path)
    fl.predict(dataset_idx=dataset_idx)


def run_gradient_inversion(args):
    fl_ginv = GradientInversion(args)
    fl_ginv.load_data()
    fl_ginv.set_nets()
    fl_ginv.inversion()


if __name__ == '__main__':

    args = get_args()
    # run_fl_training(args)
    # run_predict(dataset_idx=9)

    run_gradient_inversion(args)



