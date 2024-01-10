import cv2
import mediapipe as mp

class HandsList:
    def __int__(self):
        self.right = None
        self.left = None

    def __str__(self):
        result = ""
        if self.hasRight():
            result += str(self.right)
        else:
            result += "Right: not found. "
        if self.hasLeft():
            result += " " + str(self.left)
        else:
            result += "Left: not found. "
        return result

    def addHand(self, hand):
        if hand.side == RT:
            self.right = hand
        else:
            self.left = hand

    def count(self):
        return int(self.right is not None) + int(self.left is not None)

    def clear(self):
        self.left = None
        self.right = None

    def hasRight(self):
        return self.right is not None

    def hasLeft(self):
        return self.left is not None


