import tkinter as tk
from tkinter import filedialog
from tkinter import *
import face_recognition
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image,ImageTk
import numpy as np
from mouth_open import get_lip_height, get_mouth_height


def facialExpressionModel(json_file,weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top,background='#CDCDCD',font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('hface.xml')
mouthc = cv2.CascadeClassifier('hmouth.xml')
eyec = cv2.CascadeClassifier('hteye.xml')

model = facialExpressionModel('model_a.json','model_weights.h5')

Emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

def is_mouth_open(face_landmarks):
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)
    
    ratio = 0.5
    # print('top_lip_height: %.2f, bottom_lip_height: %.2f, mouth_height: %.2f, min*ratio: %.2f' 
    #       % (top_lip_height,bottom_lip_height,mouth_height, min(top_lip_height, bottom_lip_height) * ratio))
          
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return True
    else:
        return False

def detect(file_path):
    global Label_packed

    img = cv2.imread(file_path)

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_img,1.3,5)
    eyes = eyec.detectMultiScale(gray_img,1.3,5)
    mout='Not Detected'
    eye='Closed'
    final_image = face_recognition.load_image_file(file_path)

    face_landmarks_list = face_recognition.face_landmarks(final_image)
    try:
        for(x,y,w,h) in faces:
            fc = gray_img[y:y+h,x:x+w]
            rel = cv2.resize(fc,(48,48))
            pred = model.predict(rel[np.newaxis,:,:,np.newaxis])
            res = Emotions[np.argmax(pred)]
            print(res)

        for face_landmarks in face_landmarks_list:
            if(is_mouth_open(face_landmarks=face_landmarks)):
                mout = "Mouth Opened"
            else:
                mout = "Mouth Closed"

        for (ex,ey,ew,eh) in eyes:
            eye = 'Opened'

        label1.configure(foreground='#011638',text="Emotion :- "+res+' Eyes :- '+eye+' Mouth :- '+mout)
    except:
        label1.configure(foreground='#011638',text="Unavailabel to detect")

def show_detectButton(file_path):

    detect_b = Button(top,text="Detect Emotion",command= lambda: detect(file_path),padx=10,pady=5)
    detect_b.configure(background='#7C8E05',foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.49)


def upload_image():
    try:
        file_path = filedialog.askopenfile()
        print(file_path)
        uploaded = Image.open(file_path.name)
        uploaded.thumbnail(((top.winfo_width()/2.3),(top.winfo_height()/2.3)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image = im)

        sign_image.image = im
        label1.configure(text='')
        print('File UIploaded')
        show_detectButton(file_path.name)
        

    except:
        print("Error in Uploading")


def live_Capture():
    emotion = []
    eye = ''
    mout = "Not Detected"
    cap = cv2.VideoCapture(0)

    while True:


        succ,img = cap.read()


        global Label_packed

        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_img,1.3,5)
        eyes = eyec.detectMultiScale(gray_img,1.3,5)
        face_landmarks_list = face_recognition.face_landmarks(img)

        try:
            for(x,y,w,h) in faces:
                fc = gray_img[y:y+h,x:x+w]
                rel = cv2.resize(fc,(48,48))
                pred = model.predict(rel[np.newaxis,:,:,np.newaxis])

                index = np.argmax(pred)

                confidence = round(pred[0,index]*100, 1)
                print(confidence)

                res = Emotions[np.argmax(pred)]
                
                if confidence > 65.0:

                    if len(emotion) > 0: 
                            if Emotions[np.argmax(pred)] != emotion[-1]:
                                emotion.append(res)      
                    else:
                        emotion.append(res)       
        except:
            label1.configure(foreground='#011638',text="Unable to detect")

        if len(emotion) > 5: 
            emotion = emotion[-5:]

        for face_landmarks in face_landmarks_list:
            ret_mouth_open = is_mouth_open(face_landmarks)
            if ret_mouth_open is True:
                mout = 'Mouth Opened'
            else:
                mout = 'Mouth Closed'

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)   


        cv2.rectangle(img, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, ' '.join(emotion), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, mout, (10,90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Emotion Detector",img)

        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


upload = Button(top,text='Upload Image',command=upload_image,padx=10,pady=5)
upload.configure(background='#7C8E05',foreground='white',font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand=True)

Live = Button(top,text='Live Detection',command=live_Capture,padx=10,pady=5)
Live.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
Live.pack(side='top',pady=10)
sign_image.pack(side='bottom',expand=True)


label1.pack(side='bottom',expand=True)
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()