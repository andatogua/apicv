from tensorflow.keras.models import load_model
import cv2
import numpy as np
def get_age(distr):
    distr = distr*4
    if distr >= 0.65 and distr <= 1.4:return "0-18"
    if distr >= 1.65 and distr <= 2.4:return "19-30"
    if distr >= 2.65 and distr <= 3.4:return "31-80"
    if distr >= 3.65 and distr <= 4.4:return "80 +"
    return "Unknown"
    
def get_gender(prob):
    if prob < 0.5:return "Male"
    else: return "Female"

sex_f=['Male','Female']
model = load_model('gender_model.h5')
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gender_ranges = ['Hombre', 'Mujer']
while True:
    _, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    imageAux = gray.copy()
    for j,(x,y,w,h) in enumerate(faces):
        
        rostro = imageAux[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(100,100),interpolation = cv2.INTER_AREA)
        rostro = np.reshape(rostro,(rostro.shape[0],rostro.shape[1],1))
        val = model.predict( np.array([ rostro ]) )   
        #output_gender=gender_ranges[np.argmax(val)]
        print(j,val)
        index_male = val[0][0]
        index_female = val[0][1]
        
        index_gender = index_female / index_male

        if index_gender > 0.5:
            output_gender = 1
            color = (35,124,229)
        else:
            output_gender = 0
            color = (133,156,27)
        #age = get_age(val[0])
        #gender = get_gender(val[1])
        text = "Gender: " + str(gender_ranges[output_gender])
        #text2 =" Age: " +str(age) + " - " + str(val[0])
        cv2.putText(img,text,(x,y+h+20),cv2.FONT_HERSHEY_COMPLEX,0.5,color)
        cv2.rectangle(img, (x,y),(x+w,y+h),color,2)
        #cv2.putText(img,text2,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
    copy = img.copy()
    cv2.imshow("test",copy)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
video.release()
cv2.destroyAllWindows()


#img = cv2.imread('./test5.jpeg',0)
#img = np.reshape(img,(128,128,3))
#print(val) 

