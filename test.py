import cv2
import numpy as np
import time
from tkinter import *
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image,ImageTk
from datetime import datetime

age_model = load_model('D:\LearnByself\Python\Age\Age.h5')
age_labels = ['18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50']
age=""
#Khoi tao giao dien gui
tk=Tk()
tk.title("Age Regconition")
tk.geometry("800x500+0+0")
tk.resizable(0,0)
tk.configure(background="white")
#Hien thi ten khung hinh
lb04=Label(tk,text="CAM",font="Times 20",fg="blue",bg="white")
lb04.pack()
lb04.place(x=380,y=50)
#Hien thi so luong loai
lb11=Label(tk,fg="blue",bg="white",font="Times 25",text="Age: ")
lb11.pack()
lb11.place(x=340,y=420)


#khoi tao camera bang webcam laptop
capture = cv2.VideoCapture(0)
def close_window():
    tk.destroy()
def ConvertImage(convert_img):
    image = convert_img[:,80:(80+480)]
    image = cv2.resize(image, dsize =(128,128))
    image = np.expand_dims(image, axis=0)
    return image
def Regconition(reg_img):
    #Age
    age_predict = age_model.predict(reg_img)
    age_label= age_labels[np.argmax(age_predict)]
    return age_label
while capture.isOpened():
    ret, image_ori = capture.read()
    cv2.imwrite('image_ori.jpg',image_ori)
    imagelg=Image.open('image_ori.jpg')
    imagelg=imagelg.resize((400,300),Image.ANTIALIAS)
    imagelg=ImageTk.PhotoImage(imagelg)
    lb05=Label(image=imagelg)
    lb05.image=imagelg 
    lb05.pack()
    lb05.place(x=200,y=110)
    tk.update()
    image = ConvertImage(image_ori)
    age = Regconition(image)
    lb21=Label(tk,fg="green",bg="white",font="Times 25",text=age)
    lb21.pack()
    lb21.place(x=420,y=420)
    age = ""

cv2.destroyAllWindows()
tk.mainloop()