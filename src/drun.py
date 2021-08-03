import sys
import os
from tkinter import *
from PIL import ImageTk,Image
import tkinter as tk




window=Tk()

window.title("Driver Fatigue Detection in Vehicles using Computer Vision")

background = "background.png"
photo = PhotoImage(file = background)

label_for_image= Label(window, image=photo)

label_for_image.place(x=0, y=0, relwidth=1, relheight=1)
label_for_image.pack()
w = photo.width()
h = photo.height()
#window.geometry('%dx%d+0+0' % (w,h))
window.geometry('1024x800')
def run():
    os.system('python dcode.py')


b = Button(window, text="Initialize", command=run, height=4 , width=10,justify=CENTER,font =
('calibri', 24, 'bold'),fg='black',bg='black')
b.place(relx = 0.5, rely = 0.5,anchor = CENTER)

window.mainloop()



