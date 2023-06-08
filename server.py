# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import base64
import json
import random

import mediapipe as mp
import face_recognition
from tensorflow.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_detection
mp_facemesh = mp.solutions.face_mesh

faceClassif = cv2.CascadeClassifier('gender_model/haarcascade_frontalface_default.xml')
model = load_model('gender_model/gender_model.h5')
gender_ranges = ['Hombre', 'Mujer']
imagePaths = ['Isaac', 'JC']
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')

app = FastAPI()
app.mount("/static",StaticFiles(directory="./client/static"),name="static")
templates = Jinja2Templates(directory="./client")

@app.get("/")
def home(request:Request):
    return templates.TemplateResponse("detect.html",{"request":request})

@app.get("/login")
def home(request:Request):
    return templates.TemplateResponse("login.html",{"request":request})

@app.get("/registerface")
def home(request:Request):
    return templates.TemplateResponse("capture.html",{"request":request})


@app.websocket('/capture')
async def capture(websocket: WebSocket):
    flag = False
    validate = False
    
    act_x = False
    act_y = False
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            capture = data["capture"]
            username = data["username"]
            base64_data = data["image"]
            encoded_data = base64_data.split(',')[1]
            nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if capture and not flag:
                cv2.imwrite(f"./users/{username}.jpeg",img)
                flag = True
                
            #get roi
            height,width,c = img.shape
            mw = int(width/2)
            mh = int(height/2)
            roiy1 = mh - 30
            roiy2 = mh + 30
            roix1 = mw - 30
            roix2 = mw + 30

            with mp_face.FaceDetection(
                min_detection_confidence=0.85) as face_mesh:

                results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                annotated_image = img.copy()
                if results.detections is not None:
                    for detection in results.detections:
                        # Bounding Box
                        xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                        ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                        w = int(detection.location_data.relative_bounding_box.width * width)
                        h = int(detection.location_data.relative_bounding_box.height * height)
                        cv2.rectangle(annotated_image, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 2)
                        # right eye point
                        x_RE = int(detection.location_data.relative_keypoints[0].x * width)
                        y_RE = int(detection.location_data.relative_keypoints[0].y * height)
                        cv2.circle(annotated_image, (x_RE, y_RE), 1, (0, 0, 255), 2)
                        # left eye point
                        x_LE = int(detection.location_data.relative_keypoints[1].x * width)
                        y_LE = int(detection.location_data.relative_keypoints[1].y * height)
                        cv2.circle(annotated_image, (x_LE, y_LE), 1, (255, 0, 255), 2)
                        # nose point
                        x_NT = int(detection.location_data.relative_keypoints[2].x * width)
                        y_NT = int(detection.location_data.relative_keypoints[2].y * height)
                        cv2.circle(annotated_image, (x_NT, y_NT), 1, (255, 0, 0), 2)
                        # mouth point
                        x_MC = int(mp_face.get_key_point(detection, mp_face.FaceKeyPoint.MOUTH_CENTER).x * width)
                        y_MC = int(mp_face.get_key_point(detection, mp_face.FaceKeyPoint.MOUTH_CENTER).y * height)
                        cv2.circle(annotated_image, (x_MC, y_MC), 1, (0, 255, 0), 2)
                        # right ear point
                        x_RET = int(mp_face.get_key_point(detection, mp_face.FaceKeyPoint.RIGHT_EAR_TRAGION).x * width)
                        y_RET = int(mp_face.get_key_point(detection, mp_face.FaceKeyPoint.RIGHT_EAR_TRAGION).y * height)
                        cv2.circle(annotated_image, (x_RET, y_RET), 1, (0, 255, 255), 2)
                        # left ear point
                        x_LET = int(mp_face.get_key_point(detection, mp_face.FaceKeyPoint.LEFT_EAR_TRAGION).x * width)
                        y_LET = int(mp_face.get_key_point(detection, mp_face.FaceKeyPoint.LEFT_EAR_TRAGION).y * height)
                        cv2.circle(annotated_image, (x_LET, y_LET), 1, (255, 255, 0), 2)
                        
                        #verify activity
                        if not (roix1 < x_NT < roix2):
                            act_x = True
                        if not (roiy1 < y_NT < roiy2):
                            act_y = True
                        if act_x and act_y:
                            validate = True

            retval, buffer = cv2.imencode('.jpg', annotated_image)
            jpg_as_bytes = base64.b64encode(buffer)
            await websocket.send_json({"validate":validate,"image":jpg_as_bytes.decode('utf-8'),"save":flag})
    except WebSocketDisconnect:
        print("close")

@app.websocket("/recognize")
async def recognize(websocket:WebSocket):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # Initialize some variables
    username = None
    known_face_encodings =[]
    known_face_names = []
    face_locations = []
    face_encodings = []
    face_names = []
    detect = True

    cont = 0

    names_to_draw = ["chin","left_eyebrow","right_eyebrow","nose_bridge","nose_tip","left_eye","right_eye","top_lip","bottom_lip"]
    landmarks_to_draw = {}

    await websocket.accept()
    try:
        while True:
            validate = False

            data = await websocket.receive_text()
            data = json.loads(data)
            if username == None:
                username = data["username"]
                known_face_names.append(username)
                user_image = face_recognition.load_image_file(f"./users/{username}.jpeg")
                user_face_encoding = face_recognition.face_encodings(user_image,model="cnn")[0]
                known_face_encodings.append(user_face_encoding)

            base64_data = data["image"]
            encoded_data = base64_data.split(',')[1]
            nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if detect:
                detect = not detect
                continue
            if len(names_to_draw) > 0:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                landmarks = face_recognition.face_landmarks(rgb_small_frame,face_locations)
                face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        if known_face_names[first_match_index] == username:
                            name = known_face_names[first_match_index]
                            validate = True
                            cont += 1

                    # Or instead, use the known face with the smallest distance to the new face
                    # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    # best_match_index = np.argmin(face_distances)
                    # if matches[best_match_index]:
                    #     #verify username
                    #     if known_face_names[best_match_index] == username:
                    #         name = known_face_names[best_match_index]
                    #         validate = True

                    face_names.append(name)

                    if validate and cont % 4 == 0 and len(names_to_draw) > 0:
                        random_choice = random.choice(names_to_draw)
                        if random_choice not in landmarks_to_draw.keys():
                            landmarks_to_draw.update({random_choice:landmarks[0][random_choice]})
                        for n in landmarks_to_draw.keys():
                            landmarks_to_draw.update({n:landmarks[0][n]})
                        names_to_draw.remove(random_choice)


                        for values in landmarks_to_draw.values():
                            for x,y in values:
                                x *= 4
                                y *= 4
                                cv2.circle(img,(x,y),1,(0,255,0))
            else:
                validate =True
                with mp_facemesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5) as face_mesh:
                    # Convert the BGR image to RGB before processing.
                    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if not results.multi_face_landmarks:
                        continue
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_facemesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )



            # Display the results
            # for (top, right, bottom, left), name in zip(face_locations, face_names):
            #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            #     top *= 4
            #     right *= 4
            #     bottom *= 4
            #     left *= 4

                # Draw a box around the face
                #cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                #cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                #font = cv2.FONT_HERSHEY_DUPLEX
                #cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


            #send data
            retval, buffer = cv2.imencode('.jpg', img)
            jpg_as_bytes = base64.b64encode(buffer)
            #stringify data
            #response = json.dumps({"validate":validate,"image":jpg_as_bytes})
            detect = not detect
            await websocket.send_json({"validate":validate,"image":jpg_as_bytes.decode('utf-8')})

    except WebSocketDisconnect:
        print("close")

@app.websocket('/detect')
async def capture(websocket: WebSocket):
    output_gender = 2
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            username = data["username"]
            base64_data = data["image"]
            encoded_data = base64_data.split(',')[1]
            nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceClassif.detectMultiScale(gray, 1.3, 5)
            imageAux = gray.copy()
            auxFrame = gray.copy()
            if len(faces) == 0:
                output_gender = 2
            for j,(x,y,w,h) in enumerate(faces):
                rostro_fd = auxFrame[y:y+h,x:x+w]
                rostro_fd = cv2.resize(rostro_fd,(150,150),interpolation= cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro_fd)
                
                if result[1] < 70:
                    detect = {"detect":True,"index":result[0]}
                else:
                    detect = {"detect":False,"index":""}
                    rostro = imageAux[y:y+h,x:x+w]
                    rostro = cv2.resize(rostro,(100,100),interpolation = cv2.INTER_AREA)
                    rostro = np.reshape(rostro,(rostro.shape[0],rostro.shape[1],1))
                    val = model.predict( np.array([ rostro ]) )   
                    #output_gender=gender_ranges[np.argmax(val)]
                    index_male = val[0][0]
                    index_female = val[0][1]
                    
                    index_gender = index_female / index_male

                    if index_gender > 0.5:
                        output_gender = 1
                    else:
                        output_gender = 0
                    
            await websocket.send_json({"gender":output_gender,"detect":detect})

    except WebSocketDisconnect:
        print("close")