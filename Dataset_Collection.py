import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

                  #Green            #Orange        #Purple          #Blue          #Dark Blue     #Light green
color_theme_BGR = [(16, 245, 117), (66, 125, 255), (173, 103, 100), (255, 197, 0), (174, 132, 0), (210, 247, 228)]
color_theme_RGB = [(117, 245, 16), (255, 125, 66), (100, 103, 173), (0, 197, 255), (0, 132, 174), (228, 247, 210)]

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=color_theme_BGR[1], thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=color_theme_BGR[0], thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=color_theme_BGR[3], thickness=2, circle_radius=3),mp_drawing.DrawingSpec(color=color_theme_BGR[2], thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=color_theme_BGR[0], thickness=2, circle_radius=3),mp_drawing.DrawingSpec(color=color_theme_BGR[2], thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(image, results. right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=color_theme_BGR[0], thickness=2, circle_radius=3),mp_drawing.DrawingSpec(color=color_theme_BGR[2], thickness=2, circle_radius=1))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left_hand, left_hand])


if __name__ == '__main__':
	print('1')
	DATA_PATH = os.path.join('Dataset') 
	print('2')
	actions = np.array(['Yes'])
	print('3')
	clips_amount = 100
	frame_per_clip = 30
	start_folder = 1
	for action in actions:
		for sequence in range(start_folder, start_folder + clips_amount):
			try:
				os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
			except:
				pass

	print('4')
	cap = cv.VideoCapture(0)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, 300)
	cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
	print('5')
	mp_holistic = mp.solutions.holistic
	print('6')
	mp_drawing = mp.solutions.drawing_utils
	print('7')
	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
	    
	    for action in actions:
	        for sequence in range(start_folder, start_folder + clips_amount):
	            if sequence > clips_amount:
	                break
	                
	            for frame_counter in range(frame_per_clip):

	                ret, frame = cap.read()                
	                image, results = mediapipe_detection(frame, holistic)
	                draw_landmarks(image, results)
	                
	                image = cv.flip(image, 1)

	                if frame_counter == 0: 
	                    cv.putText(image, 'Press any button', (60,200), cv.FONT_HERSHEY_SIMPLEX, 3, color_theme_BGR[4], 6, cv.LINE_AA)
	                    cv.putText(image, f' {action}: {sequence}', (15,25), cv.FONT_HERSHEY_SIMPLEX, 0.75, color_theme_BGR[1], 1, cv.LINE_AA)
	                    
	                    cv.imshow('HHv0.3.4-Dataset_Collection', image)
	                    cv.waitKey(0)
	                    
	                else: 
	                    cv.putText(image, f' {action}: {sequence}', (15,25), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_theme_BGR[0], 1, cv.LINE_AA)               
	                    cv.imshow('HHv0.3.4-Dataset_Collection', image)

	                #cv.resizeWindow('HHv0.3.4-Dataset_Collection', 600, 400)

	                keypoints = extract_keypoints(results)
	                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_counter))
	                try:
	                    np.save(npy_path, keypoints)
	                except:
	                    break
	                
	                if cv.waitKey(10) & 0xFF == ord('z'):
	                    break
	                    
	    cap.release()
	    cv.destroyAllWindows()

