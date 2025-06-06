import argparse
import json
import torch
import random
import numpy as np
import os
import textwrap
from configs import add_parser
from utils.dataloader import load_data
from long_term_forecasting import LTF_Trainer, loss_funcs

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def print_args(args):
    args_str = str(args)
    wrapped_args = textwrap.fill(args_str, width=100, break_long_words=False, subsequent_indent='    ')
    print(wrapped_args)


def save_args(params, file_path):
    with open(file_path, 'w') as f:
        json.dump(params, f, indent=4)


def load_args(file_path):
    with open(file_path, 'r') as f:
        args = argparse.Namespace(**json.load(f))
    return args


def run_experiment(args, task, setting):
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('>>>>>> Args in experiment: <<<<<<')
    print_args(args)

    data, corr = load_data(args)
    engine = LTF_Trainer(args, task, setting, corr)

    if args.use_multi_gpu:
        engine.model = torch.nn.DataParallel(engine.model, device_ids=args.device_ids)

    if args.is_training == 1:
        engine.train(data=data)
        torch.cuda.empty_cache()
        vali_loss = engine.validate(vali_loader=data['val_loader'], loss_func=loss_funcs[args.loss])
        return vali_loss


    if args.is_training == 2:
        print(f"Testing with parameters: learning_rate={args.learning_rate}, d_conv={args.d_conv}, d_state={args.d_state}")
        mse = engine.test(test_loader=data['test_loader'])
        return mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiMamba')
    add_parser(parser=parser)
    args, model_name = parser.parse_known_args()
    args.model = 'BiMamba'

    if torch.cuda.is_available() and not args.use_cpu:
        args.device = 'cuda:{}'.format(args.gpu)
        print(f"Using single GPU: {args.device}")
        if args.use_multi_gpu:
            args.device_ids = [int(i) for i in args.device_ids.split(',')]
            print(f"Using multiple GPUs: {args.device_ids}")
    else:
        args.device = 'cpu'
        print("Using CPU")

    task = '{}({})_{}_{}_loss({})'.format(
        args.dataset_path,
        args.task,
        args.seq_len,
        args.pred_len,
        args.loss
    )

    setting = '{0}+batch_size{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}'.format(
        args.model,
        args.batch_size,
        f'+e_layers{args.e_layers}' if hasattr(args, 'e_layers') else '',
        f'+(d_model{args.d_model}+d_ff{args.d_ff})' if hasattr(args, 'd_model') else '',
        f'+n_heads{args.n_heads}' if hasattr(args, 'n_heads') else '',
        f'+(d_conv{args.d_conv}+d_state{args.d_state})' if hasattr(args, 'd_conv') else '',
        f'+dropout{args.dropout:.2f}' if hasattr(args, 'dropout') else '',
        f'+(patch_len{args.patch_len}+stride{args.stride})' if hasattr(args, 'patch_len') else '',
        '', '', ''
    )

    # 使用 args 中的默认值进行实验
    if args.is_training == 1:
        print("Starting training with default parameters.")
        run_experiment(args, task, setting)
        # print(f"Validation Loss: {vali_loss}")

    if args.is_training == 2:
        print("Starting testing with default parameters.")
        mse = run_experiment(args, task, setting)
        print(f"Test MSE: {mse}")
