# import cv2
# import mediapipe as mp
# import time
#
# class handdetector():
#     def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):
#         self.mode=mode
#         self.maxhands=maxhands
#         self.detectioncon=detectioncon
#         self.trackcon=trackcon
#         self.mphands=mp.solutions.hands
#         self.hands= self.mphands.Hands(self.mode, self.maxhands, self.detectioncon, self.trackcon)
#         self.mpdraw=mp.solutions.drawing_utils
#
#
#     def findhands(self,img,draw=True):
#         imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         results=self.hands.process(imgRGB)
#         if results.multi_hand_landmarks:
#             for handlms in results.multi_hand_landmarks:
#                 if draw:
#                     self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
#         return img
#                 # for id,lm in enumerate(handlms.landmark):
#                 #
#                 #     h,w,c=img.shape
#                 #     cx,cy=int(lm.x*w),int(lm.y*h)
#                 #     print(id,cx,cy)
#                 #         # if id==4:
#                 #     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
#
#
#
# def main():
#     ptime = 0
#     ctime = 0
#     cap = cv2.VideoCapture(0)
#     detector =  handdetector()
#
#     while True:
#         success, img = cap.read()
#         img = detector.findhands(img)
#         ctime = time.time()
#         fps = 1 / (ctime - ptime)
#         ptime = ctime
#
#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
#
#         cv2.imshow("image", img)
#         cv2.waitKey(1)
#
#
#
#
#
#
# if  __name__ =="__main__":
#     main()
import cv2
import mediapipe as mp
import time
import math

class handdetector():
    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            self.mode,
            self.maxhands,
            min_detection_confidence=self.detectioncon,
            min_tracking_confidence=self.trackcon
        )
        self.mpdraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
        return img

    def findposition(self, img, handno=0,draw=True):
        lmlist=[]

        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handno]



            for id,lm in enumerate(myhand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])

                # cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmlist

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handdetector()


    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findhands(img)
        lmlist = detector.findposition(img,draw=False)
        if len(lmlist) != 0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
