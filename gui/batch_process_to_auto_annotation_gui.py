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

# add labels to tabs
tab_control.add(settings_tab, text='Settings')
tab_control.add(advanced_settings_tab, text='Advanced')
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

# create placeholder in the advanced settings tab
lbl_binary_threshold = ttk.Label(advanced_settings_tab, text='Double Trigger Threshold').grid(column=0, row=0)
binary_threshold = tk.StringVar()
binary_threshold.set('.804')
ent_binary_threshold = ttk.Entry(advanced_settings_tab, textvariable=binary_threshold).grid(column=1, row=0, sticky='w')
lbl_binary_threshold_2 = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=0, sticky='w')

lbl_reverse_trigger_threshold = ttk.Label(advanced_settings_tab, text='Reverse Trigger Threshold').grid(column=0, row=1)
reverse_trigger_threshold = tk.StringVar()
reverse_trigger_threshold.set('4.8e-02')
ent_reverse_trigger_threshold = ttk.Entry(advanced_settings_tab, textvariable=reverse_trigger_threshold).grid(column=1, row=1, sticky='w')
lbl_reverse_trigger_threshold__2 = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=1, sticky='w')

lbl_premature_termination_threshold = ttk.Label(advanced_settings_tab, text='Premature Termination Threshold').grid(column=0, row=2)
premature_termination_threshold = tk.StringVar()
premature_termination_threshold.set('3.2e-02')
ent_premature_termination_threshold = ttk.Entry(advanced_settings_tab, textvariable=premature_termination_threshold).grid(column=1, row=2, sticky='w')
lbl_premature_termination_threshold_2 = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=2, sticky='w')

lbl_flow_undershoot_threshold = ttk.Label(advanced_settings_tab, text='Flow Undershoot Threshold').grid(column=0, row=3)
flow_undershoot_threshold = tk.StringVar()
flow_undershoot_threshold.set('0.71')
ent_flow_undershoot_threshold = ttk.Entry(advanced_settings_tab, textvariable=flow_undershoot_threshold).grid(column=1, row=3, sticky='w')
lbl_flow_undershoot_threshold_2 = ttk.Label(advanced_settings_tab, text='[0-1]').grid(column=2, row=3, sticky='w')


# function that will grab all user input and pass it to function
def pass_user_input():

    import_path_input = import_path.get()
    export_path_input = export_path.get()
    exe_filepath_input = exe_path.get()
    generate_triplets_and_statics_input = generate_triplets_and_statics.get()
    generate_annotations_input = generate_annotations.get()

    # get threshold inputs, convert strings to float, set min to 0, set max to 1
    binary_threshold_input = min(max(float(binary_threshold.get()), 0), 1)
    reverse_trigger_threshold_input = min(max(float(reverse_trigger_threshold.get()), 0), 1)
    premature_termination_threshold_input = min(max(float(premature_termination_threshold.get()), 0), 1)
    flow_undershoot_threshold_input = min(max(float(flow_undershoot_threshold.get()), 0), 1)

    # turnthresholds to dict
    binary_thresholds = {'Double Trigger': binary_threshold_input}

    multilabel_thresholds = {'Double Trigger Reverse Trigger': reverse_trigger_threshold_input,
                            'Double Trigger Premature Termination': premature_termination_threshold_input,
                            'Double Trigger Flow Undershoot': flow_undershoot_threshold_input}


    batch_process_to_auto_annotation.main(
        import_path_input,
        export_directory=export_path_input,
        vent_annotator_filepath=exe_filepath_input,
        generate_triplets_and_statics=generate_triplets_and_statics_input,
        generate_annotations=generate_annotations_input,
        binary_threshold=binary_thresholds,
        multitarget_thresholds=multilabel_thresholds
    )

# create button that will use arguments from entries and run function
button1 = tk.Button(text='Click me Run', command=pass_user_input).pack()

app_window.mainloop()
