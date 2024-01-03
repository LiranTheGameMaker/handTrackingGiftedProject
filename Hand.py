from enum import Enum

class Hand():
    def __init__(self):
        self.cx = 0
        self.cy = 0
        self.pre_x = 0
        self.pre_y = 0
        self.name = ""
        self.certainty = 0
        self.landmarks = None

    def Update(self, classification, landmarks):
        self.name = classification.classification[0].label
        self.landmarks = landmarks

    def get_center_landmark(self):
        return 9



