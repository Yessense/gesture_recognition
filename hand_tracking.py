import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2)

width = 640
height = 480


class HandDetector():
    def __init__(self, mode: bool = False,
                 max_hands: int = 2,
                 detection_con: float = 0.5,
                 track_con: float = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
                                         max_num_hands=max_hands,
                                         min_detection_confidence=detection_con,
                                         min_tracking_confidence=track_con)

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw: bool = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand: int = 0, draw: bool = True):
        landmarks_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                landmarks_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

        return landmarks_list


def main():
    width = 640
    height = 480

    p_time: float = 0.
    c_time: float = 0.

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img, draw=False)
        landmarks_list = detector.find_position(img, draw=True)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
