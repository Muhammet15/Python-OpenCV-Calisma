import cv2
import os
import numpy as np
from PIL import Image
import json

yol = 'dataset'

tani = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


def getImagesAndLabels(yol):
    faceSamples=[]
    ids=[]
    labels=[]
    klasorler=os.listdir(yol)
    dictionary = {}
    for i,kl in enumerate(klasorler):
        dictionary[kl]=int(i)
    f = open("ids.json","w")
    a = json.dump(dictionary,f)
    f.close()
    
    for kl in klasorler:
        for res in os.listdir(os.path.join(yol,kl)):
            PIL_img = Image.open(os.path.join(yol,kl,res)).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            id = int(dictionary[kl])
            faces =detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
    return faceSamples,ids

faces,ids =getImagesAndLabels(yol)
tani.train(faces,np.array(ids))
tani.write('trainer.yml')
