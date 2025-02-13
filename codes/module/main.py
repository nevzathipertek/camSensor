#--------------------------------------------------------------------
#program::sample code for Camera Module
# @uthor::Nevzat DURMAZ
#   date::16.05.2024
# update::10.09.2024
#version::01.03
#--------------------------------------------------------------------
import sensor, image, time, os, tf, uos, gc, pyb
from OLED import OLED
from RELAY import RELAY
from LEDS import LEDS
from BUTTONS import BUTTONS

led = pyb.LED(3)
led.on() #turning red led on

relay = RELAY()
#relay.turn(False)

leds = LEDS(duty = 48, freq = 180000)

bt = BUTTONS(nTimeCheck = 8)

sensor.reset()  # reset cam
sensor.set_pixformat(sensor.GRAYSCALE)  # GRAYSCALE / RGB565

# QQVGA:160x120 / QVGA:320x240 / VGA:640x480 / B128X64: 128x64 / B128X128: 128x128 / WVGA2:752x480
sensor.set_framesize(sensor.QVGA)  # frame size
sensor.set_hmirror(False)
sensor.set_vflip(False)
#sensor.set_transpose(True)
sensor.set_auto_exposure(True, exposure_us=4000)  #set exposure time
#sensor.set_auto_exposure(True, exposure_us=400)  #set exposure time
sensor.set_auto_gain(False) # AGC enable
sensor.skip_frames(time=2000)  # wait for adjustment



clock = time.clock()

usb = pyb.USB_VCP()


ol = OLED()
ol.set_textsize(2)


led.off() #turning red led off
#while (usb.isconnected()==False):

while True:

    bt.checkButtons()

    if bt.leftBut():
        relay.turn(True)
    else:
        relay.turn(False)

    if bt.rightBut():
        print('Right ON ')
    else:
        print('Right OFF ')

    clock.tick()
    img = sensor.snapshot()


    print(clock.fps(), "fps")


