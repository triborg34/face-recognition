





import random
import cv2
import requests


def insertToPocketBase(name,face_image,frame,score):
        randomNumber=random.randint(1,10000)
        cv2.imwrite(f"images/face/image_{name}_{randomNumber}.jpg", face_image)
        cv2.imwrite(f"images/frame/frame_{name}_{randomNumber}.jpg", frame)
        file1=open(f"images/face/image_{name}_{randomNumber}.jpg", "rb")
        file2=open(f"images/frame/frame_{name}_{randomNumber}.jpg", "rb")
        files={
                "image": (f"images/face/image_{name}_{randomNumber}.jpg", file1, "image/jpeg"),
                "frame": (f"/imagesframe/frame_{name}_{randomNumber}.jpg", file2, "image/jpeg"),
                     }
                                    
        res=requests.post("http://127.0.0.1:8090/api/collections/faces/records",data={"name":name,"confidence":score,},files=files)
        print(res.json())
        
    