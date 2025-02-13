#program::BUTTONS
# @uthor::Nevzat DURMAZ
#   date::16.05.2024
# update::16.05.2024
#version::01.01

from pyb import Pin

class BUTTONS():

    def __init__(self, nTimeCheck = 20):

        self.nTimeCheck = nTimeCheck
        self.leftCount = 0
        self.rightCount = 0
        self.leftButton = Pin(Pin('P2'), Pin.IN)
        self.rightButton = Pin(Pin('P3'), Pin.IN)
        self.leftButtonPosition = False
        self.rightButtonPosition = False

    def buttonLeftInstant(self):
        return False if self.leftButton.value()==True else True

    def buttonRightInstant(self):
        return False if self.rightButton.value()==True else True

    # put this method for checking buttons position
    def checkButtons(self):

        if self.leftButtonPosition:
            if self.leftButton.value():
                self.leftCount = 0
            else:
                self.leftCount += 1
                if self.leftCount > self.nTimeCheck:
                    self.leftButtonPosition = False
        else:
            if self.leftButton.value():
                self.leftCount += 1
                if self.leftCount > self.nTimeCheck:
                    self.leftButtonPosition = True
            else:
                self.leftCount = 0

        if self.rightButtonPosition:
            if self.rightButton.value():
                self.rightCount = 0
            else:
                self.rightCount += 1
                if self.rightCount > self.nTimeCheck:
                    self.rightButtonPosition = False
        else:
            if self.rightButton.value():
                self.rightCount += 1
                if self.rightCount > self.nTimeCheck:
                    self.rightButtonPosition = True
            else:
                self.rightCount = 0

    def leftBut(self):
        return self.leftButtonPosition

    def rightBut(self):
        return self.rightButtonPosition

"""
bt = BUTTONS(nTimeCheck = 8)

while True:
    bt.checkButtons()
    if bt.leftBut():
        ol.show_string(0, 2, 'Left ON ')
    else:
        ol.show_string(0, 2, 'Left OFF')
"""
