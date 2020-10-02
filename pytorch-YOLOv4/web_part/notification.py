import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os
def insertButton(root,text,width,side,expand,font,highlightthickness,bg,fg,command,padx,pady,ipadx,ipady,enterColor,bd = 0):
    button = Button(root,text = text, width = width,highlightthickness = highlightthickness,bd = bd,command = command)
    button.pack(side = side, fill = BOTH, expand = expand,padx = padx, pady = pady, ipadx = ipadx,ipady = ipady)
    button.config(font=font, bg=bg, fg=fg)
    def on_enter_side(e):
        button['bg'] = enterColor
    def on_leave_side(e):
        button['bg'] = bg
    button.bind("<Enter>", on_enter_side)
    button.bind("<Leave>", on_leave_side)
    return button
def createFrame(root,width,height,side,background,expand,padx,pady,ipadx,ipady):
    frame = Frame(root, width=width, height=height, background=background)
    frame.pack(side = side, fill = BOTH, expand = expand, padx = padx, pady = pady, ipadx = ipadx,ipady = ipady)
    return frame
def insertText(root,value,width,height,side,bg,fg,expand,font,sticky,padx,pady,ipadx,ipady):
    text = Label(root, text = value,width = width,height = height)
    text.pack(side = side, expand = expand, fill = BOTH,anchor = sticky,padx = padx, pady = pady, ipadx = ipadx,ipady = ipady) 
    text.config(font=font,bg=bg,fg=fg)
def login(vehicle_type,number, gateNumber):
    # if(root):
    #     root.destroy()
    root = tk.Tk(className=' Alert Message')
    root.geometry("500x300+760+230")
    root.resizable(0, 0)
    root.config(bg = 'white')
    font = 'Roboto'
    upperFrame = createFrame(root,500,50,TOP,'#FFFFFF',True,0,0,0,0)
    bottomFrame = createFrame(root,500,250,TOP,'#FFFFFF',True,0,0,0,0)
    # signinFrame = createFrame(middleFrame,500,200,TOP,'#FFFFFF',True,0,0,0,0)
    image = Image.open("./images/falcon.jpeg")
    image = image.resize((100,100),Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    panel = Label(upperFrame, image = img, background = "white")
    panel.pack(side = TOP, fill = "both", expand = False)
    insertText(bottomFrame,'A ' + vehicle_type + " has been passed throug gate no " + str(gateNumber) + "\n Please take Some Action",10,1,TOP,'#FFFFFF','#000000',True,(font,16),W,0,0,0,0)
    insertButton(bottomFrame,'Report',12,TOP,False,(font,10),0,'#455A64','#FFFFFF',lambda: Report(root,number,vehicle_type,gateNumber),200,5,0,0,'#FFFFFF')
    insertButton(bottomFrame,'Ignore',12,TOP,False,(font,10),0,'#455A64','#FFFFFF',lambda: Ignore(root),200,5,0,0,'#FFFFFF')
    root.mainloop()
def Report(root,number,vehicle_type,gateNumber):
    # give notification to nearby police
    if(root):
        root.destroy()
    return None
def Ignore(root):
    # ignore the alert
    if(root):
        root.destroy()
    return None
