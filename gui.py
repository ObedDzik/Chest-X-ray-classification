#!/usr/bin/env python
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
#from tensorflow.keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input

#load trained model to classify the images
model = load_model('best_model.hdf5')

#Initialize GUI

top=tk.Tk()
top.geometry('800x600')
top.title('Chest Xray Image Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path).convert('RGB')
    image = image.resize((224,224))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    img_preprocessed = preprocess_input(image)
    prediction = model.predict(img_preprocessed)
    lbl = 'Diseased' if prediction>0.5 else 'Healthy'
    print(lbl)
    label.configure(foreground='#011638', text=lbl)

def show_classify_button(file_path):
    classify_button = Button(top, text = 'Classify Image',
    command=lambda: classify(file_path),padx=10, pady=5)
    classify_button.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))
    classify_button.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image", command = upload_image, padx=10, pady=5)

upload.configure(background='#364156', foreground='white', font=('arial',10,'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Chest Xray Image Classification", pady=20, font=('arial', 20, 'bold'))

heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
