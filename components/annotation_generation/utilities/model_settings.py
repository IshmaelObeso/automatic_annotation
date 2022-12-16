from pathlib import Path

""" 

This file contains information about models we want to use in our program, including paths, name, and output columns

"""

# model paths
double_trigger_model = '.\\models\\double_trigger_model.onnx'
auto_trigger_model = '.\\models\\autotrigger_model.onnx'
delayed_termination_model = '.\\models\\delayed_termination_model.onnx'
flow_undershoot_model = '.\\models\\flow_undershoot_model.onnx'
premature_termination_model = '.\\models\\premature_termination_model.onnx'
reverse_trigger_model = '.\\models\\reverse_trigger_model.onnx'

double_trigger_model_path = Path(double_trigger_model)
auto_trigger_model_path = Path(auto_trigger_model)
delayed_termination_model_path = Path(delayed_termination_model)
flow_undershoot_model_path = Path(flow_undershoot_model)
premature_termination_model_path = Path(premature_termination_model)
reverse_trigger_model_path = Path(reverse_trigger_model)

MODELS_DICT = {
    'Double Trigger': {'path': double_trigger_model_path,
                       'use': True,
                       'output_columns': ['Double Trigger'],
                       'threshold': {'Double Trigger': .9},
                       'dyssynch_code': 107,
                       'channel': 'AirwayPressure',
                       },
    'Autotrigger': {'path': auto_trigger_model_path,
                     'use': True,
                    'output_columns': ['Autotrigger'],
                    'threshold': {'Autotrigger': .9},
                    'dyssynch_code': 104,
                    'channel': 'SpirometryFlow',
                       },
    'Delayed Termination': {'path': delayed_termination_model_path,
                            'use': True,
                       'output_columns': ['Delayed Termination'],
                       'threshold': {'Delayed Termination': .9},
                       'dyssynch_code': 110,
                       'channel': 'SpirometryFlow',
                       },
    'Flow Undershoot': {'path': flow_undershoot_model_path,
                        'use': True,
                       'output_columns': ['Flow Undershoot'],
                       'threshold': {'Flow Undershoot': .9},
                       'dyssynch_code': 109,
                       'channel': 'SpirometryFlow',
                       },
    'Premature Termination': {'path': premature_termination_model_path,
                              'use': True,
                       'output_columns': ['Premature Termination'],
                       'threshold': {'Premature Termination': .9},
                       'dyssynch_code': 111,
                       'channel': 'SpirometryFlow',
                       },
    'Reverse Trigger': {'path': reverse_trigger_model_path,
                        'use': True,
                       'output_columns': ['Reverse Trigger'],
                       'threshold': {'Reverse Trigger': .9},
                       'dyssynch_code': 114,
                       'channel': 'SpirometryFlow',
                       },

}