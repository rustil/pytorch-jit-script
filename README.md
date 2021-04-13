# Pytorch to TorchScript

This minimal script basically is a small wrapper around `torch.jit.script(model)`, so that it uses the JIT script to convert a python model to a torch script model that is consequently saved.

The script needs the implementation of the model in a `*.py` python file and the name of the class that is the model implementation. Optionally, a set of weights can be loaded to the model, which will then be saved alongside the model in the TorchScript file. 
In order to do the Just-In-Time conversion of the python implementation, the model class is instantiated and any necessary arguments to the constructor of the model can be passed via the `--args` and `--kwargs` flags.

For a full list of flags, please consult the `-h` option:

```
usage: pytorch_to_torch.py [-h] [-s SAVE_PATH] [-w STATE_DICT]
                           [--args ARGS [ARGS ...]]
                           [--kwargs KWARGS [KWARGS ...]] [--version]
                           module_path model_name

positional arguments:
  module_path           Path to the "`*.py`" python implementation of the
                        PyTorch model that is to be converted to TorchScript
  model_name            Name of the model (Python class name) that is to be
                        converted.

optional arguments:
  -h, --help            show this help message and exit
  -s SAVE_PATH, --save_path SAVE_PATH
                        Path where the converted model should be saved to.
                        (default: ./torchScriptModel.pt)
  -w STATE_DICT, --state_dict STATE_DICT
                        Path to the state dict / checkpoint that is to be
                        loaded. (default: None)
  --args ARGS [ARGS ...]
                        Positional arguments for the model constructor
                        (default: [])
  --kwargs KWARGS [KWARGS ...]
                        Keyword arguments for the model constructor. Has to be
                        an even number of arguments in <key> <val> <key> <val>
                        ... fashion. (default: {})
  --version             show program's version number and exit

```

For the `--args` and `--kwargs`, input is parsed as a `str` if it isn't convertible to a `float` or `int`. Given that it is a numeric value, it is converted to `float` if it contains a `.` and to `int` otherwise.

The script has been tested with PyTorch 1.8.0.

## Caveats

Please remember that not all functionality of pytorch is convertible using the `jit.script` approach, which will result in an error and which either needs a change for the model or maybe an update of pytorch. The Pytorch API documentation usually warns you when a method or class is not convertible.

