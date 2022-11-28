import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import *
import sys

sys.path.append('..')

from scripts import batch_process_to_auto_annotation

# set color of gui
background_color = '#87d6d0'

# create main application window
app_window = tk.Tk()
# create title for main window
app_window.title('Bennotator')
# make application resizable
app_window.resizable(width=True, height=True)

# define style
style = ttk.Style()
style.theme_use('alt')

# define tabs
tab_control = ttk.Notebook(app_window)
settings_tab = ttk.Frame(tab_control)
advanced_settings_tab = ttk.Frame(tab_control)
filter_settings_tab = ttk.Frame(tab_control)

# add labels to tabs
tab_control.add(settings_tab, text='Settings')
tab_control.add(advanced_settings_tab, text='Advanced')
tab_control.add(filter_settings_tab, text='Filter Settings')
tab_control.pack(expand=1, fill='both')

# create buttons and labels in settings tab

# functions to browse for files
def browse_import_button():
    global import_path
    folder_name = filedialog.askdirectory()
    import_path.set(folder_name)

def browse_export_button():
    global export_path
    folder_name = filedialog.askdirectory()
    export_path.set(folder_name)

def browse_exe_button():
    global exe_path
    file_name = filedialog.askopenfilename()
    exe_path.set(file_name)

def browse_filter_button():
    global filter_filepath
    file_name = filedialog.askopenfilename()
    filter_filepath.set(file_name)

## Settings Tab
# create import path label and entry field
lbl_import_path = ttk.Label(settings_tab, text='Import Directory').grid(column=0, row=0)
import_path = tk.StringVar()
btn_import_folder_path = ttk.Button(settings_tab, text='Browse', command=browse_import_button).grid(column=1, row=0, sticky='w')
ent_import_path = ttk.Entry(settings_tab, textvariable=import_path, width=50).grid(column=2, row=0, sticky='w')

# create export path label and entry field
lbl_export_path = ttk.Label(settings_tab, text='Export Directory').grid(column=0, row=1)
export_path = tk.StringVar()
btn_export_folder_path = ttk.Button(settings_tab, text='Browse', command=browse_export_button).grid(column=1, row=1, sticky='w')
ent_export_path = ttk.Entry(settings_tab, textvariable=export_path, width=50).grid(column=2, row=1, sticky='w')

# create exe path label and entry field
lbl_exe_path = ttk.Label(settings_tab, text='Batch Processor Exe Filepath').grid(column=0, row=2)
exe_path = tk.StringVar()
exe_path.set('.\\batch_annotator\RipVent.BatchProcessor.exe')
btn_exe_filepath = ttk.Button(settings_tab, text='Browse', command=browse_exe_button).grid(column=1, row=2, sticky='w')
ent_exe_path = ttk.Entry(settings_tab, textvariable=exe_path, width=50).grid(column=2, row=2, sticky='w')

# create label and checkbox for generating triplets and statics
lbl_generate_triplets_and_statics = ttk.Label(settings_tab, text='Generate Triplets and Statics').grid(column=0, row=3)
generate_triplets_and_statics = BooleanVar(value=True)
ent_generate_triplets_and_statics = ttk.Checkbutton(settings_tab, variable=generate_triplets_and_statics).grid(column=1, row=3, sticky='w')

# create label and checkbox for generating annotations
lbl_generate_annotations = ttk.Label(settings_tab, text='Generate Annotations').grid(column=0, row=4)
generate_annotations = BooleanVar(value=True)
ent_generate_annotations = ttk.Checkbutton(settings_tab, variable=generate_annotations).grid(column=1, row=4, sticky='w')

# create label and checkbox for deleting triplets and spectral triplets after generating statics
lbl_delete_triplets = ttk.Label(settings_tab, text='Delete Triplets and Spectral Triplets after generating statics').grid(column=0, row=5)
delete_triplets = BooleanVar(value=False)
ent_delete_triplets = ttk.Checkbutton(settings_tab, variable=delete_triplets).grid(column=1, row=5, sticky='w')

## Advanced Settings Tab
# create label and entry field for Double Trigger Threshold
lbl_double_trigger_threshold = ttk.Label(advanced_settings_tab, text='Double Trigger Threshold').grid(column=0, row=0)
double_trigger_threshold = tk.StringVar()
double_trigger_threshold.set('.5')
ent_double_trigger_threshold = ttk.Entry(advanced_settings_tab, textvariable=double_trigger_threshold).grid(column=1, row=0, sticky='w')
lbl_double_trigger_threshold_info = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=0, sticky='w')

# create label and entry field for Auto Trigger Threshold
lbl_auto_trigger_threshold = ttk.Label(advanced_settings_tab, text='Auto Trigger Threshold').grid(column=0, row=1)
auto_trigger_threshold = tk.StringVar()
auto_trigger_threshold.set('.5')
ent_auto_trigger_threshold = ttk.Entry(advanced_settings_tab, textvariable=auto_trigger_threshold).grid(column=1, row=1, sticky='w')
lbl_auto_trigger_threshold_info = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=1, sticky='w')

# create label and entry field for Delayed Termination Threshold
lbl_delayed_termination_threshold = ttk.Label(advanced_settings_tab, text='Delayed Termination Threshold').grid(column=0, row=2)
delayed_termination_threshold = tk.StringVar()
delayed_termination_threshold.set('.5')
ent_delayed_termination_threshold = ttk.Entry(advanced_settings_tab, textvariable=delayed_termination_threshold).grid(column=1, row=2, sticky='w')
lbl_delayed_termination_threshold_info = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=2, sticky='w')

# create label and entry field for Flow Undershoot Threshold
lbl_flow_undershoot_threshold = ttk.Label(advanced_settings_tab, text='Flow Undershoot Threshold').grid(column=0, row=3)
flow_undershoot_threshold = tk.StringVar()
flow_undershoot_threshold.set('.5')
ent_flow_undershoot_threshold = ttk.Entry(advanced_settings_tab, textvariable=flow_undershoot_threshold).grid(column=1, row=3, sticky='w')
lbl_flow_undershoot_threshold_info = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=3, sticky='w')

# create label and entry field for Premature Termination Threshold
lbl_premature_termination_threshold = ttk.Label(advanced_settings_tab, text='Premature Termination Threshold').grid(column=0, row=4)
premature_termination_threshold = tk.StringVar()
premature_termination_threshold.set('.5')
ent_premature_termination_threshold = ttk.Entry(advanced_settings_tab, textvariable=premature_termination_threshold).grid(column=1, row=4, sticky='w')
lbl_premature_termination_threshold_info = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=4, sticky='w')

# create label and entry field for Reverse Trigger Threshold
lbl_reverse_trigger_threshold = ttk.Label(advanced_settings_tab, text='Reverse Trigger Threshold').grid(column=0, row=5)
reverse_trigger_threshold = tk.StringVar()
reverse_trigger_threshold.set('.5')
ent_reverse_trigger_threshold = ttk.Entry(advanced_settings_tab, textvariable=reverse_trigger_threshold).grid(column=1, row=5, sticky='w')
lbl_reverse_trigger_threshold_info = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=5, sticky='w')

## Filter Settings Tab
# create label and checkbox for generating triplets and statics
lbl_filter_file = ttk.Label(filter_settings_tab, text='Use Excel File for Filtering').grid(column=0, row=1)
use_filter_file = BooleanVar(value=False)
ent_filter_file = ttk.Checkbutton(filter_settings_tab, variable=use_filter_file).grid(column=1, row=1, sticky='w')

# create browse button to input filtering file
lbl_filter_browse = ttk.Label(filter_settings_tab, text='Filtering File Path').grid(column=0, row=2)
filter_filepath = tk.StringVar()
btn_filter_filepath = ttk.Button(filter_settings_tab, text='Browse', command=browse_filter_button).grid(column=1, row=2, sticky='w')
ent_filter_path = ttk.Entry(filter_settings_tab, textvariable=filter_filepath, width=50).grid(column=2, row=2, sticky='w')

# create

# function that will grab all user input and pass it to function
def pass_user_input():

    import_path_input = import_path.get()
    export_path_input = export_path.get()
    exe_filepath_input = exe_path.get()
    generate_triplets_and_statics_input = generate_triplets_and_statics.get()
    generate_annotations_input = generate_annotations.get()
    delete_triplets_input = delete_triplets.get()

    # get threshold inputs, convert strings to float, set min to 0, set max to 1
    double_trigger_threshold_input = min(max(float(double_trigger_threshold.get()), 0), 1)
    auto_trigger_threshold_input = min(max(float(auto_trigger_threshold.get()), 0), 1)
    delayed_termination_threshold_input = min(max(float(delayed_termination_threshold.get()), 0), 1)
    flow_undershoot_threshold_input = min(max(float(flow_undershoot_threshold.get()), 0), 1)
    premature_termination_threshold_input = min(max(float(premature_termination_threshold.get()), 0), 1)
    reverse_trigger_threshold_input = min(max(float(reverse_trigger_threshold.get()), 0), 1)

    # turn thresholds to dict
    threshold_dict = {}
    threshold_dict['Double Trigger'] = {'Double Trigger': double_trigger_threshold_input}
    threshold_dict['Autotrigger'] = {'Autotrigger': auto_trigger_threshold_input}
    threshold_dict['Delayed Termination'] = {'Delayed Termination': delayed_termination_threshold_input}
    threshold_dict['Flow Undershoot'] = {'Flow Undershoot': flow_undershoot_threshold_input}
    threshold_dict['Premature Termination'] = {'Premature Termination': premature_termination_threshold_input}
    threshold_dict['Reverse Trigger'] = {'Reverse Trigger': reverse_trigger_threshold_input}

    # get filter inputs
    use_filter_input = use_filter_file.get()
    filter_filepath_input = filter_filepath.get()

    # make dict for filter info
    filter_info = {}
    filter_info['use'] = use_filter_input
    filter_info['filepath'] = filter_filepath_input
    # TODO: Figure out how to best have user pass this infomation
    filter_info['exclude_columns_and_values'] = {'Reviewed by:': 'NaN', 'analysis exclusion': 'not NaN'}



    batch_process_to_auto_annotation.main(
        import_path_input,
        export_directory=export_path_input,
        vent_annotator_filepath=exe_filepath_input,
        generate_triplets_and_statics=generate_triplets_and_statics_input,
        generate_annotations=generate_annotations_input,
        thresholds_dict=threshold_dict,
        filter_file_info=filter_info,
        delete_triplets_and_spectral_triplets=delete_triplets_input
    )

# create button that will use arguments from entries and run function
button1 = tk.Button(text='Click me Run', command=pass_user_input).pack()

app_window.mainloop()
