from tkinter import *
import tkinter as tk
import os
from shutil import rmtree, move
from tkinter import messagebox, filedialog
import argparse
import os
from functools import partial
from tkinter.ttk import *
import sys

sys.path.append('..')

from scripts import batch_annotate_generate_triplets_and_statics


# set color of gui
background = '#87d6d0'

# creates a gui window, background color and class name
root = tk.Tk(className = 'annotate breaths')
root['background']=background
canvas1 = tk.Canvas(root, width=400, height=340)
canvas1['background']=background
canvas1.pack()

# create title for gui and place it in the gui
title = tk.Label(root, text='Batch Annotate Breaths \nAKA Dissync Headstone')
title.config(font=('helvetica bold', 14))
title['background']=background
canvas1.create_window(200, 25, window=title)

# create titles for different entries into gui
input_types = ['Import Path:', 'Export Path:']
# empty list to save entries that are put into gui
entries = []
# plament of where the entries will be placed in gui (vertically)
placement_y = 50

# loop through inputs
for input in input_types:
    # for the first entry point we want to start higher up with a smaller jump than the entries that follow
    if input == 'Import Path:':
        placement_y += 20
    else:
        placement_y += 40

    # create title for entry input as well as its placement and background color
    input_title = tk.Label(root, text= input)
    input_title.config(font=('helvetica', 10))
    input_title['background'] = background
    canvas1.create_window(200, placement_y, window=input_title)

    # create a window for user to input entry
    entry = tk.Entry(root, width = 50)
    placement_y += 30
    canvas1.create_window(200, placement_y, window=entry)

    # save entry from gui input
    entries.append(entry)

# import our image to use as a run button (lol)
# photo = PhotoImage(file='..\\faces.png')

# creating a label for our run button
Label(root, text="Generate Triplets and Statics",font=('helvetica 13')).pack(pady=10)

# create button that runs the batch annotate function above when clicked
# button1 = tk.Button(image=photo, command=batch_annotate, borderwidth=0).pack()
# if we dont want keiths face as a button heres a boring option...
# input_entry = entries[0].get()
# export_entry = entries[1].get()

button1 = tk.Button(text='Click me to Batch Annotate', command=lambda:
     batch_annotate_generate_triplets_and_statics.main(entries[0].get(), entries[1].get()), bg='brown', fg='white').pack()

# create a window for our button so it'll always fit no matter how many inputs we create
canvas1.create_window(200, 180, window=button1)

root.mainloop()