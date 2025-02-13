#program::RELAY
# @uthor::Nevzat DURMAZ
#   date::16.05.2024
# update::16.05.2024
#version::01.01
from pyb import LED

class RELAY():

    def __init__(self, relayState = False):

        self.relay = LED(4)
        self.relayState = relayState
        self.relaySetPositionDirectly(self.relayState)

    def relaySetPositionDirectly(self, state = False):
        if state == True:
            self.relay.off()
        else:
            self.relay.on()

    def turn(self, state = False):
        if self.relayState != state:
            self.relayState = state
            if self.relayState == True:
                self.relay.off()
            else:
                self.relay.on()



"""
rl = RELAY()
rl.turn(True)
"""
