import cv2
import os


cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
face_detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

global user
user = input("Kullanıcı ismi giriniz ...: ")

print("\n kameraya bakin ")

say = 0
os.mkdir('dataset/'+user)


while(True):
    ret,cerceve = cam.read()
    cerceve = cv2.flip(cerceve,1)
    gri = cv2.cvtColor(cerceve,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gri,1.5,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(cerceve,(x,y),(x+w,y+h),(255,0,0),2)
        say +=1
        path ="dataset/"+user+"/"
        cv2.imwrite(path+str(say)+".jpg",gri[y:y+h,x:x+w])
        cv2.imshow('DATA',cerceve)
    k= cv2.waitKey(100) & 0xff
    if k ==27:
        break
    elif say>=50:
        break

cam.release()
cv2.destroyAllWindows()