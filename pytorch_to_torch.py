#!/usr/bin/env python
# coding: utf-8
import argparse
import importlib.util
import os
from collections import OrderedDict

import torch

print('PyTorch version: {}'.format(torch.__version__))

def make_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('module_path',
                        type=str,
                        help='Path to the "`*.py`" python implementation of the PyTorch model that is to be converted \
                        to TorchScript')

    parser.add_argument('model_name',
                        type=str,
                        help='Name of the model (Python class name) that is to be converted.')

    parser.add_argument('-s', '--save_path',
                        type=str,
                        default='./torchScriptModel.pt',
                        help='Path where the converted model should be saved to.')

    parser.add_argument('-w', '--state_dict',
                        type=str,
                        help='Path to the state dict / checkpoint that is to be loaded.')
    parser.add_argument('--args', nargs='+',
                        help='Positional arguments for the model constructor',
                        default=[]
                        )
    parser.add_argument('--kwargs', nargs='+',
                        help='Keyword arguments for the model constructor. Has to be an even number of arguments in \
                        <key> <val> <key> <val> ... fashion.',
                        default={})

    parser.add_argument('--version', action='version', version='PyToTorchScript 0.01')

    return parser


def execute(parser):
    # Import python implementation
    name = os.path.basename(parser.module_path)[:-3]
    spec = importlib.util.spec_from_file_location(name, parser.module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module is None:
        raise ImportError('"module" has not been imported properly. Please check {} is a valid python file.'.format(
            parser.module_path))

    # Instantiate model and potentially load state dictionary
    model = getattr(module, parser.model_name)(*parser.args, **parser.kwargs)

    if parser.state_dict is not None:
        checkpoint = torch.load(parser.state_dict)
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print('State dictionary might have been saved using DataParallel, trying key-renaming hack now...')
            # Hack because module was saved using DataParallel
            # (https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/2)
            checkpoint = OrderedDict({k.replace("module.", ""): v for k, v in checkpoint.items()})
            model.load_state_dict(checkpoint)

        print('State dictionary successfully loaded.')
    else:
        print('No state dictionary is loaded.')

    script_model = torch.jit.script(model)

    print(' Python implementation '.center(80, '-'))
    print(script_model.code)
    print(' TorchScript implementation '.center(80, '-'))
    print(script_model)

    script_model.save(parser.save_path)


def converter(x):
    if "." in x:
        try:
            return float(x)
        except ValueError:
            return x
    else:
        try:
            return int(x)
        except ValueError:
            return x


def main():
    parser = make_parser()
    parser_namespace = parser.parse_args()

    if len(parser_namespace.kwargs) > 0:
        if len(parser_namespace.kwargs) == 0 or len(parser_namespace.kwargs) % 2 != 0:
            raise ValueError(
                '--kwargs arguments set with an odd number of arguments -> {}'.format(parser_namespace.kwargs))
        else:
            parser_namespace.kwargs = {k: converter(v) for k, v in
                                       zip(parser_namespace.kwargs[::2], parser_namespace.kwargs[1::2])}

    parser_namespace.args = [converter(x) for x in parser_namespace.args]

    if not os.path.exists(parser_namespace.module_path):
        raise FileNotFoundError(
            'Implementation file not found at: {}. Maybe try an absolute path.'.format(parser_namespace.module_path))

    print(parser_namespace)
    execute(parser_namespace)


if __name__ == "__main__":
    main()
