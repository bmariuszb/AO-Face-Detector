from glob import glob
from sys import platform
import cv2
from multiprocessing import cpu_count
from tensorflow import config, keras
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, _tkinter_finder
import os


def video(root):
    global frame2, cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror('Error', 'Nie wykryto kamery!')
        cap.release()
        return 0
    frame2 = tk.Frame(root, bg="black", width=800, height=650)
    frame2.pack()
    label = tk.Label(frame2)
    label.grid(row=0, column=0)

    def show_frames():
        ret, frame = cap.read()  
        if not ret:
            messagebox.showerror('Error', 'Odlaczono kamere!')
            cap.release()
            global cameraOn
            cameraOn=False
            return 0
        thicknessx=int(0.01*frame.shape[0])
        thicknessy=int(0.01*frame.shape[1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if(len(faces) > 0):
            for (x, y, w, h) in faces:
                img = frame[y:y + h, x:x + w]
                try:
                    array = np.array([cv2.resize(img, (128, 128))]) / 255

                    result = model.predict(array)
                    if result[0][0] > 0.9:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thicknessx)
                    else:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thicknessy)
                except:
                    print('not found')

        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgg = Image.fromarray(imageRGB)
        imgtk = ImageTk.PhotoImage(image=imgg)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(1, show_frames)
    show_frames()
    return 1


def start():
    global win, cameraOn, frame2
    good = 1
    if not cameraOn:
        if frame2:
            frame2.destroy()
        good = video(win)
    if good:
        cameraOn = True
    else:
        cameraOn = False


def stop():
    global cameraOn, frame2
    if cameraOn:
        frame2.destroy()
        cap.release()
    cameraOn = False


def obrazek():
    filepath = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title="Wybierz obrazek",
                                          filetypes= (("PNG","*.png"),
                                          ("JPG","*.jpg"), ("JPEG","*.jpeg")))
    if filepath:
        global win, frame2, cameraOn
        if frame2:
            frame2.destroy()
            if cameraOn:
                cap.release()
                cameraOn=False
        
        cv_img = cv2.imread(filepath)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        thicknessx=int(0.01*cv_img.shape[0])
        thicknessy=int(0.01*cv_img.shape[1])
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if(len(faces) > 0):
            for (x, y, w, h) in faces:
                img = cv_img[y:y + h, x:x + w]
                try:
                    array = np.array([cv2.resize(img, (128, 128))]) / 255

                    result = model.predict(array)
                    if result[0][0] > 0.9:
                        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), thicknessx)
                    else:
                        cv2.rectangle(cv_img, (x, y), ( x +w, y + h), (0, 0, 255), thicknessy)
                except:
                    print('not found')

        imageRGB = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height = 500 * imageRGB.shape[0] / imageRGB.shape[1]
        imageRGB = cv2.resize(imageRGB, (500, int(height)))

        frame2 = tk.Frame(win, bg="black", width=500, height=int(height))
        frame2.pack()
        label = tk.Label(frame2)
        label.grid(row=0, column=0)
        
        imgg = Image.fromarray(imageRGB)
        imgtk = ImageTk.PhotoImage(image=imgg)
        label.imgtk = imgtk
        label.configure(image=imgtk)


if __name__ == '__main__':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    config.threading.set_inter_op_parallelism_threads(cpu_count())
    path = os.path.dirname(os.path.realpath(__file__))
    if platform == 'win32':
        var = '\\'
    else:
        var = '/'
    model = keras.models.load_model(path + var + 'MODEL')
    faceCascade = cv2.CascadeClassifier(path + var + 'haarcascade_frontalface_default.xml')

    win = tk.Tk()
    win.title('Analiza Obrazów - projekt')
    win.geometry("800x700")
    win.configure(bg="#333333")
    cameraOn = False
    frame2 = 0
    cap = 0

    button_frame = tk.Frame(win, bg='#333333')
    button_frame.pack()
    tk.Button(button_frame, width=22, pady=5, relief=tk.RAISED, bd=10, bg="#6b9494",text="START",command=start).pack(side='left', padx=25)
    tk.Button(button_frame, width=22, pady=5, relief=tk.RAISED, bd=10, bg="#6b9494",text="STOP",command=stop).pack(side='left', padx=25, pady=50)
    tk.Button(button_frame, width=22, pady=5, relief=tk.RAISED, bd=10, bg="#6b9494",text="WCZYTAJ OBRAZ",command=obrazek).pack(side='left', padx=25)

    win.mainloop()
