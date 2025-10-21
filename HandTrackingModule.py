import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands = 2, model_complexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
    
        # Go through all hands and hand landmarks to get positions of all points
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    
    def findPosition(self, img, handNum=0, draw=True):
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            
            for id, lm in enumerate(myHand.landmark):
                # Landmark values are given in ratios of the img, so we need to convert them into pixel values
                h, w, c = img.shape # Height, width, channels
                cx, cy = int(lm.x * w), int (lm.y * h) # Pixel position

                lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,0), cv2.FILLED)
        
        return lmList
    

# This main() code can be copied into different projects to use HandTracking as an insertable module
def main():
    cap = cv2.VideoCapture(0) # Webcam number
    
    pTime = 0 # Previous Time
    cTime = 0 # Current Time
    
    detector = HandDetector()

    while True:
        success, original_img = cap.read()
        img = cv2.flip(original_img, 1) # Flipping the camera feed horizontally so it looks correct
        
        # Find hands
        img = detector.findHands(img)
        
        # Find position for only one hand
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            print(lmList[4]) # Thumb tip position
        
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10,35), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        # Show image
        cv2.imshow("Flipped Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()