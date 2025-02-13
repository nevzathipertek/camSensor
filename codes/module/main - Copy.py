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

#leds = LEDS(duty = 88, freq = 180000)

bt = BUTTONS(nTimeCheck = 8)

sensor.reset()  # reset cam
sensor.set_pixformat(sensor.GRAYSCALE)  # GRAYSCALE / RGB565

# QQVGA:160x120 / QVGA:320x240 / VGA:640x480 / B128X64: 128x64 / B128X128: 128x128 / WVGA2:752x480
sensor.set_framesize(sensor.B128X64)  # frame size
sensor.set_hmirror(False)
sensor.set_vflip(False)
#sensor.set_transpose(True)
sensor.set_auto_exposure(True, exposure_us=4000)  #set exposure time
#sensor.set_auto_exposure(True, exposure_us=400)  #set exposure time
sensor.set_auto_gain(False) # AGC enable
sensor.skip_frames(time=2000)  # wait for adjustment

#net = tf.load_builtin_model("model1.tflite")
#net = tf.load("model1.tflite", load_to_fb=True)

net = None
labels = None

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')


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

#    if bt.rightBut():
#        print('Right ON ')
#    else:
#        print('Right OFF ')

    clock.tick()
    img = sensor.snapshot()

    output = net.classify(img)

    if output:

        max_idx = output[0][0].index(max(output[0][0]))
        print("Sınıf:", labels[max_idx], "İhtimal:", output[0][0][max_idx])

    img.draw_string(10, 10, labels[max_idx], color=(255, 0, 0), scale=2)

    print(clock.fps(), "fps")


