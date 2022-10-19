import tkinter as tk
from tkinter import ttk
from tkinter import *
import sys

sys.path.append('..')

from scripts import batch_process_to_auto_annotation


# set color of gui
background_color = '#87d6d0'

# # creates a gui window, background color and class name
# root = tk.Tk(className = 'annotate breaths')
# root['background']=background
# canvas1 = tk.Canvas(root, width=400, height=340)
# canvas1['background']=background
# canvas1.pack()

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

# create import path label and entry field
lbl_import_path = ttk.Label(settings_tab, text='Import Directory').grid(column=0, row=0)
import_path = tk.StringVar()
ent_import_path = ttk.Entry(settings_tab, textvariable=import_path, width=50).grid(column=1, row=0)

# create export path label and entry field
lbl_export_path = ttk.Label(settings_tab, text='Export Directory').grid(column=0, row=1)
export_path = tk.StringVar()
ent_export_path = ttk.Entry(settings_tab, textvariable=export_path, width=50).grid(column=1, row=1)

# create exe path label and entry field
lbl_exe_path = ttk.Label(settings_tab, text='Batch Processor Exe Filepath').grid(column=0, row=2)
exe_path = tk.StringVar()
exe_path.set('.\\batch_annotator\RipVent.BatchProcessor.exe')
ent_exe_path = ttk.Entry(settings_tab, textvariable=exe_path, width=50).grid(column=1, row=2)

# create label and checkbox for generating triplets and statics
lbl_generate_triplets_and_statics = ttk.Label(settings_tab, text='Generate Triplets and Statics').grid(column=0, row=3)
generate_triplets_and_statics = BooleanVar(value=True)
ent_generate_triplets_and_statics = ttk.Checkbutton(settings_tab, variable=generate_triplets_and_statics).grid(column=1, row=3, sticky='w')

# create label and checkbox for generating annotations
lbl_generate_annotations = ttk.Label(settings_tab, text='Generate Annotations').grid(column=0, row=4)
generate_annotations = BooleanVar(value=True)
ent_generate_annotations = ttk.Checkbutton(settings_tab, variable=generate_annotations).grid(column=1, row=4, sticky='w')

# create placeholder in the advanced settings tab
lbl_placeholder = ttk.Label(advanced_settings_tab, text='Placeholder, Tab will include options for generating predictions and settings thresholds in the future.\n Dont touch unless you read the paper').grid(column=0, row=0)





# function that will grab all user input and pass it to function
def pass_user_input():

    import_path_input = import_path.get()
    export_path_input = export_path.get()
    exe_filepath_input = exe_path.get()
    generate_triplets_and_statics_input = generate_triplets_and_statics.get()
    generate_annotations_input = generate_annotations.get()

    batch_process_to_auto_annotation.main(
        import_path_input,
        export_directory=export_path_input,
        vent_annotator_filepath=exe_filepath_input,
        generate_triplets_and_statics=generate_triplets_and_statics_input,
        generate_annotations=generate_annotations_input
    )

    print('---------Done!---------')
# create button that will use arguments from entries and run function
button1 = tk.Button(text='Click me Run', command=pass_user_input).pack()


#
# # create title for gui and place it in the gui
# title = tk.Label(root, text='Auto Annotate Breaths')
# title.config(font=('helvetica bold', 14))
# title['background']=background
# canvas1.create_window(200, 25, window=title)
#
# # create dictionary of entries and desired properties
#
#
#
# # create titles for different entries into gui
# input_types = ['Import Path:', 'Export Path:']
# # empty list to save entries that are put into gui
# entries = []
# # plament of where the entries will be placed in gui (vertically)
# placement_y = 50
#
# # loop through inputs
# for input in input_types:
#     # for the first entry point we want to start higher up with a smaller jump than the entries that follow
#     if input == 'Import Path:':
#         placement_y += 20
#     else:
#         placement_y += 40
#
#     # create title for entry input as well as its placement and background color
#     input_title = tk.Label(root, text= input)
#     input_title.config(font=('helvetica', 10))
#     input_title['background'] = background
#     canvas1.create_window(200, placement_y, window=input_title)
#
#     # create a window for user to input entry
#     entry = tk.Entry(root, width = 50)
#     placement_y += 30
#     canvas1.create_window(200, placement_y, window=entry)
#
#     # save entry from gui input
#     entries.append(entry)
#
# # import our image to use as a run button (lol)
# # photo = PhotoImage(file='..\\faces.png')
#
# # creating a label for our run button
# Label(root, text="Generate Annotations",font=('helvetica 13')).pack(pady=10)
#
# # create button that runs the batch annotate function above when clicked
# # button1 = tk.Button(image=photo, command=batch_annotate, borderwidth=0).pack()
# # if we dont want keiths face as a button heres a boring option...
# # input_entry = entries[0].get()
# # export_entry = entries[1].get()
#

# # create a window for our button so it'll always fit no matter how many inputs we create
# canvas1.create_window(200, 180, window=button1)

app_window.mainloop()
