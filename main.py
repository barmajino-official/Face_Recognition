#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import cv2
import numpy as np
import face_recognition as fr

faces_encoding_option = {}
cap = cv2.VideoCapture(1)


# In[4]:


faces_encoding_option = {}

img1 = fr.load_image_file(".\\data\\data_config\\img\\2.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
face_location1 = fr.face_locations(img1)[0]

face_encode1 = fr.face_encodings(img1)[0]



name_list = ["Barmajino","Unknown"]
cv2.rectangle(
    img1,
    (face_location1[3],face_location1[0]),
    (face_location1[1],face_location1[2]),
    (255, 0, 255)
    ,2
)


while True:

    success, img2 = cap.read()
    face_encode2 = fr.face_encodings(img2)
    face_location2 = fr.face_locations(img2)
    h = 30
    for face_location2_,face_encode2_ in zip(face_location2, face_encode2):

        result = fr.compare_faces([face_encode1],face_encode2_)
        print("result : " ,result)
        face_distance = fr.face_distance([face_encode1],face_encode2_)
        for ii in range(len(result)):
            if(result[ii]):
                name = name_list[ii]
            else:
                name = name_list[1]

        
        if(result[0]):
            color = (0,225,0)
        else:
            color = (225,0,0)
        cv2.rectangle(
            img2,
            (face_location2_[3],face_location2_[0]),
            (face_location2_[1],face_location2_[2]),
            color,
            2
        )

        
        cv2.putText(
            img2,
            f"{name} | {int(np.round(face_distance[0],2)*100)}%",
            (face_location2_[3],int(face_location2_[0]-15)),#(50,h),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            color,
            2
        )
        h += 100
    #cv2.imshow("img1",img1)
    cv2.imshow("img2",img2)
    cv2.waitKey(1)

