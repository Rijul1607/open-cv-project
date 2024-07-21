import cv2
from cvzone.HandTrackingModule import HandDetector


cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1000)

detector=HandDetector(detectionCon=0.8)

class Button():
    def __init__(self,pos,text,size=[85,85]):
        self.pos=pos
        self.size=size
        self.text=text

    def draw(self,img):
        x,y=self.pos
        w,h=self.size
        cv2.rectangle(img, self.pos,(x+w,y+h) , (255, 0, 255), cv2.FILLED)
        cv2.putText(img, self.text,(x+20 ,y+65),cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)
        return img




mybutton=Button([100,100],"Q")
mybutton1=Button([300,100],"E")
mybutton2=Button([400,100],"W")
while True:
    success,img=cap.read()


    img=mybutton.draw(img)
    img=mybutton1.draw(img)
    img=mybutton2.draw(img)

    if not success:
        break  # Break the loop if the frame is not captured correctly

        # Find hands in the frame
    hands, img = detector.findHands(img)  # Modify here to get hands information
    if hands:
        # Get the first hand detected
        hand1 = hands[0]
        lmList = hand1['lmList']  # List of 21 Landmark points
        bbox = hand1['bbox']  # Bounding box info x,y,w,h
        centerPoint = hand1['center']  # center of the hand cx,cy
        handType = hand1['type']  # Hand type Left or Right

        # Print or use the landmarks as needed
        print(lmList, bbox, centerPoint, handType)






    cv2.imshow("img",img)
    cv2.waitKey(1)