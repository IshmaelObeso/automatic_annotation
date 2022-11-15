from pathlib import Path

""" This file contains information about models we want to use in our program, including paths, name, and output columns"""

# model paths
dc_model_path = '.\\models\\dc_model.onnx'
multitarget_model_path = '.\\models\\mt_all_model.onnx'

dc_model_path = Path(dc_model_path)
multitarget_model_path = Path(multitarget_model_path)


MODELS_DICT = {
    'Binary Double Trigger': {'path': dc_model_path,
                              'output_columns': ['Double Trigger'],
                              },
    'Multi-Target': {'path': multitarget_model_path,
                        'output_columns': ['Double Trigger Reverse Trigger',
                                           'Double Trigger Premature Termination',
                                           'Double Trigger Flow Undershoot'],
                     }
}