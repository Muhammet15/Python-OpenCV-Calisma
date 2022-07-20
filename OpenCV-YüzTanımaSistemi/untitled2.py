import cv2
import os
import numpy as np
from PIL import Image
import json


tani = cv2.face.LBPHFaceRecognizer_create()
tani.read('trainer.yml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
font=cv2.FONT_HERSHEY_SIMPLEX
id=0
dictionary={}
names=[]
dosya=open("ids.json","r")
dictionary=json.load(dosya)
cam = cv2.VideoCapture(0+ cv2.CAP_DSHOW)

for key,value in dictionary.items():
    names.append(key)

while(True):
    ret,cerceve = cam.read()
    cerceve = cv2.flip(cerceve,1)
    gri = cv2.cvtColor(cerceve,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gri,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(cerceve,(x,y),(x+w,y+h),(0,255,0),2)
        id ,oran =tani.predict(gri[y:y + h , x:x + w])
        if (oran<70):
            id = names[id]
        else:
            id ="bilinmiyor"
        cv2.putText(cerceve, str(id),(x+5,y-5),font,1,(255,255,255),2)
        
        cv2.imshow("KAMERA",cerceve)
        k=cv2.waitKey(10) &0xff
        if k ==27:
            break
cam.release()
cv2.destroyAllWindows()