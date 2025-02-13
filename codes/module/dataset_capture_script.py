#--------------------------------------------------------------------
#program::sample code for Camera Module
# @uthor::Nevzat DURMAZ
#   date::16.05.2024
# update::10.09.2024
#version::01.03
#--------------------------------------------------------------------
import pyb, sensor, image, time, machine
from pyb import Pin, LED, UART, Timer

from OLED import OLED
from RELAY import RELAY
from LEDS import LEDS
from BUTTONS import BUTTONS

led = pyb.LED(3)
led.on() #turning red led on

relay = RELAY()
#relay.turn(False)

leds = LEDS(duty = 40, freq = 180000)

bt = BUTTONS(nTimeCheck = 8)

sensor.reset()  # reset cam
sensor.set_pixformat(sensor.GRAYSCALE)  # GRAYSCALE / RGB565

# QQVGA:160x120 / QVGA:320x240 / VGA:640x480 / B128X64: 128x64 / B128X128: 128x128 / WVGA2:752x480
sensor.set_framesize(sensor.B128X64)  # frame size
#sensor.set_windowing((96, 96))
sensor.set_hmirror(True)
sensor.set_vflip(True)
#sensor.set_transpose(True)
sensor.set_auto_exposure(True, exposure_us=400)  #set exposure time
sensor.set_auto_gain(False) # AGC enable
sensor.skip_frames(time=2000)  # wait for adjustment

clock = time.clock()

photoNum = 0

led.off() #turning red led off

while(True):
    bt.checkButtons()
    clock.tick()
    fps =clock.fps()
    img = sensor.snapshot()

    if bt.rightBut():
        photoNum += 1
        print(photoNum)
        #img.save("C:\Users\nevza\OneDrive\Desktop\deneme\{photoNum}.bmp")



