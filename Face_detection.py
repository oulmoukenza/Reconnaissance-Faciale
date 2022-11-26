

import cv2 
from random import randrange
import numpy as np

reponse=True

while reponse==True:
    print("Welcome!: please select one of the options below\n 1-face recognition through image\n 2-through a video \n 3-live webcam")
    print("Votre reponse: ")
    answer= input()

    Face_detection_trained=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#load trained model
    
    if answer=='3':
        
        #Lire image
        #img=cv2.imread("pic1.jpg")
        #lire video ou webcam
        webcam=cv2.VideoCapture(0)
        while True:
            #lire chaque frame dans la video
            check, frame=webcam.read()
            #Converter la couleur pour la detection
            
            grayscale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            #detecter le visage une liste 
            face_coordenates=Face_detection_trained.detectMultiScale(grayscale_img)
            # draw
            for (x,y,w,h)in face_coordenates:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(255), randrange(255), randrange(255)),5)
            
            #Affichage
            #inverser la video (mirror original)
            flip = cv2.flip(frame,1)
            #combined_window = np.hstack([frame,flip])
            cv2.imshow('face detection',flip)
            key=cv2.waitKey(1)
            if key==113 or key==81: #ord(normal 'q')
                break
        webcam.release()
        cv2.destroyAllWindows()
    elif answer=='1':
        print("Please type your image's name")
        image=input()
        img=cv2.imread(image+".jpeg")
        grayscale_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_coordenates=Face_detection_trained.detectMultiScale(grayscale_img)
        for (x,y,w,h)in face_coordenates:
            cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(255), randrange(255), randrange(255)),4)
        
        #combined_window = np.hstack([frame,flip])
        cv2.imshow('face detection',img)
        key=cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif answer=='2':
        print("Please type your video's name ")
        video_name=input()
        print("Please type your video's extension ")
        extention=input()
        video=cv2.VideoCapture(video_name+"."+extention)
        while True:
            #lire chaque frame dans la video
            check, frame=video.read()
            #Converter la couleur pour la detection
            
            grayscale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            #detecter le visage une liste 
            face_coordenates=Face_detection_trained.detectMultiScale(grayscale_img)
            # draw x=top left corner, y=bottom right corner, w=width, h=height
            for (x,y,w,h)in face_coordenates:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(255), randrange(255), randrange(255)),5)
            
            #Affichage
            #inverser la video (mirror original)
            #flip = cv2.flip(frame,1)
            #combined_window = np.hstack([frame,flip])
            cv2.imshow('face detection',frame)
            key=cv2.waitKey(1)
            if key==113 or key==81: #ord(normal 'q')
                break
        cv2.destroyAllWindows()
    else:
        print("Veuillez inserer une valeur correct entre 1 et 3 ")
    print("Vous voulez continuez ?:\n 1-oui\n 2-non")
    a=input()
    if a=='2':
        reponse=False
        print("Goodbye ^_^\nSee you again")
       