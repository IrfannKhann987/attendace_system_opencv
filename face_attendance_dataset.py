import cv2
import numpy as np
import os
import pickle

cap=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier(r'I:\onr\new_onr\new_onr\haarcascade_frontalface_default.xml')
face_data=[]
i=0
name=input('Enter your name janeman => ')


while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        crop_image=frame[y:y+h,x:x+w,:]
        resize=cv2.resize(crop_image,(50,50))
        face_data.append(resize)
        if len(face_data) <= 100 and i % 100 == 0:
            cv2.putText(frame,str(len(face_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,230),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,230),1)
    cv2.imshow('dhoprda',frame)
    k=cv2.waitKey(1)
    if k == ord('q') or len(face_data) >= 100:
        break

cap.release()
cv2.destroyAllWindows
    

face_data=np.array(face_data)
face_data=face_data.reshape(100,-1)

if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
else:
    with open('data/names.pkl','rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl','wb') as f:
        pickle.dump(names, f)


if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl','wb')as f:
        pickle.dump(face_data,f)
else:
    with open('data/face_data.pkl','rb')  as f:
        faces=pickle.load(f)
    faces=np.append(faces,face_data,axis=0)
    with open('data/face_data.pkl','wb') as f:
        pickle.dump(faces,f)


    