import numpy as np
import cv2 as cv
import mediapipe as mp
import os
from matplotlib import pyplot as plt
import time
import pyttsx3
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model as load_model

def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

color_theme = [(16, 245, 117), (66, 125, 255), (173, 103, 100), (174, 197, 0), (210, 247, 228)]
#(16, 245, 117) green
#(255, 125, 66) Orange
#(100, 103, 173) purple
#(0, 197, 255) blue
#(0, 132, 174) dark blue
#(228, 247, 210) light green
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=color_theme[1], thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=color_theme[0], thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=color_theme[3], thickness=2, circle_radius=3),mp_drawing.DrawingSpec(color=color_theme[2], thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=color_theme[0], thickness=2, circle_radius=3),mp_drawing.DrawingSpec(color=color_theme[2], thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(image, results. right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=color_theme[0], thickness=2, circle_radius=3),mp_drawing.DrawingSpec(color=color_theme[2], thickness=2, circle_radius=1))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left_hand, left_hand])

def is_hands_up(results):
    if not (results.right_hand_landmarks or results.left_hand_landmarks):
        return False
    return True
    
def render_predictions(prediction, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(prediction):
        cv.rectangle(output_frame, (0, 60 + num*40), (200, 90+num*40), color_theme[4], -1)
        cv.rectangle(output_frame, (0, 60 + num*40), (int(prob*200), 90+num*40), color_theme[0], -1)
        cv.putText(output_frame, actions[num], (0, 85+num*40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
    
    return output_frame

if __name__ == '__main__':
	actions = np.array(['Thank_you','Hello','Good'])
	model_path = os.path.join('AI_models', 'actions-v0.3.4-3 signs(Thank_you, hello, good).h5')
	model = load_model(model_path)

	print('setting up variables and cap...')
	last_30_frames = []
	detections = []
	predictions = []
	threshold = 0.7

	cap = cv.VideoCapture(0, cv.CAP_DSHOW)

	cap.set(cv.CAP_PROP_FRAME_HEIGHT, 300)
	cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)

	#print('doing mediapipe...')
	mp_holistic = mp.solutions.holistic
	mp_drawing = mp.solutions.drawing_utils

	#print('text to speech...')
	engine = pyttsx3.init()

	#print('with mp holistic...')
	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
	    frame_counter = 0
	    while True:
	        frame_counter += 1
	        has_predicted = False
	        
	        #print('cap.read() ...')
	        ret, frame = cap.read()
	        #print('mediapipe_detection...')
	        image, results = mediapipe_detection(frame, holistic)
	        #print('drawing landmarks')
	        draw_landmarks(image, results)
	        
	        #print('extracting keypoints')
	        key_points = extract_keypoints(results)
	        last_30_frames.append(key_points)
	        last_30_frames = last_30_frames[-30:]
	        #print(is_hands_up(results))
	        prediction = [0, 0, 0]
	        if len(last_30_frames) >= 30 and is_hands_up(results):
	            #print('model prediction')
	            prediction = model.predict(np.expand_dims(last_30_frames, axis = 0))[0]
	            frame_counter = 0
	            has_predicted = True
	            #print(actions[np.argmax(prediction)])
	            predictions.append(np.argmax(prediction))
	        
	        image = cv.flip(image, 1)
	        #print('visualizing')
	        try:
	            if np.unique(predictions[-10:])[0]==np.argmax(prediction):         
	                if prediction[np.argmax(prediction)] > threshold:
	                    if len(detections) > 0:
	                        if actions[np.argmax(prediction)] != detections[-1]:
	                            detections.append(actions[np.argmax(prediction)])
	                            engine.say(actions[np.argmax(prediction)])
	                            engine.runAndWait()
	                            engine.stop()
	                    else:
	                        detections.append(actions[np.argmax(prediction)])
	                        engine.say(actions[np.argmax(prediction)])
	                        engine.runAndWait()
	                        engine.stop()

	                    if len(detections) > 5:
	                        detections = detections[-5:]
	        except:
	            print('empty sequence')
	            
	        image = render_predictions(prediction, actions, image)
	        
	        cv.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
	        cv.putText(image, ' '.join(detections), (3, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
	        
	        cv.imshow('HHv0.3.4-AIRv2-5signs', image)
	        
	        if cv.waitKey(10) & 0xFF == ord('q'):
	            break

	    cap.release()
	    cv.destroywindow('HHv0.3.4-AIRv2-5signs')