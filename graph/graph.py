import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

import os

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

root = Tk.Tk()
root.wm_title("Launcher distances")

f = Figure(figsize=(20, 10), dpi=70)
a = f.add_subplot(111)


file = open("../selfMadeCarRacing/history_racing.txt", "r")
data = file.readlines()[:1000]
file.close()

data2 = []
for el in data:
    nrs = []
    rida = el.split(",")
    for nr in rida:
        nrs.append(float(nr))
    data2.append(nrs)

data = np.array(data2)

episodes = np.array(range(len(data)))
scores = data[:,1]




a.plot(episodes, scores)
#
a.set_xlabel("Episodes", fontsize=22)
a.set_ylabel("Scores", fontsize=22)
#
# # a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def _quit():
    root.quit()
    root.destroy()

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()
