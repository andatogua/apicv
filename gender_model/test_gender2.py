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
#model = load_model('Age_Sex_detection.h5')
model = load_model('gender_model.h5')
#model = load_model('Best_model_params.h5')
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    _, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    imageAux = img.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = imageAux[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(48,48))
        print(rostro.shape)
        rostro = np.reshape(rostro,(48,48,3))
        val = model.predict( np.array([ rostro ]) )   
        age = get_age(val[0])
        gender = get_gender(val[1])
        #text = "Gender: " + str(gender) + " - " + str(val[1])
        #text2 =" Age: " +str(age) + " - " + str(val[0])
        text = str(val[0]) + " - " + str(val[1])
        cv2.putText(img,text,(x,y-40),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
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

