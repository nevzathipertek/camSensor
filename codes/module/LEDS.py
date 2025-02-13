#program::LEDS (with pwm)
# @uthor::Nevzat DURMAZ
#   date::16.05.2024
# update::16.05.2024
#version::01.01

from pyb import Timer, Pin

class LEDS():

    def __init__(self, duty = 60, freq = 92000):
        #self.freq = freq
        self.duty = duty
        self.tim = Timer(4, freq = freq)
        self.pin = Pin(Pin('P7'), mode=Pin.ALT, alt=1)
        self.sPwm = self.tim.channel(1, Timer.PWM, pin=self.pin)
        self.sPwm.pulse_width_percent(self.duty)


    def seyDuty(self, duty = 60):
        if self.duty != duty:
            self.duty = duty
            self.sPwm.pulse_width_percent(self.duty)


"""
ld = LEDS(duty=0)
ld.seyDuty(50)

while True:
    pass
"""
