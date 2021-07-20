import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image
# Import libraries
import RPi.GPIO as GPIO
import time

# Set GPIO numbering mode



# Set pins 11 & 13 as outputs, and define as PWM servo1 & servo2



def load_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor


def capture_image():
    def ping():
        GPIO.setmode(GPIO.BOARD)
        # Pins connected to the HC-SR04 sensor
        TRIG = 16
        ECHO = 12
    
    
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
    
        GPIO.output(TRIG, False)
        print("waiting to settle")
        time.sleep(0.5)
    
    
        GPIO.output(TRIG, True)
        time.sleep(0.0001)
        GPIO.output(TRIG, False)
    
        while GPIO.input(ECHO) == 0:
            fPulseStart = time.time()
    
        while GPIO.input(ECHO) == 1:
            fPulseEnd = time.time()
    
        fPulseDuration = fPulseEnd - fPulseStart
    
        fDistance = fPulseDuration * 17150
    
        print("Distance:", fDistance, "cm.")
        
        if fDistance < 30:
            time.sleep(5)
            print("Distance:", fDistance, "cm.")
            if fDistance < 30:
                return True
        
        time.sleep(0.5)
    
        GPIO.cleanup()
        
        
            
    while True:
        stop = ping()
        if stop:
            break

    return True








cameraOn = True

while cameraOn:
    webcam = cv2.VideoCapture(0)
    while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = capture_image()
        
        if key == True: 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            print("Resizing image to 224x224 scale...")
            img_ = cv2.resize(img_,(224,224))
            print("Resized...")
            img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")
            
    
            new_model = tf.keras.models.load_model('mbnetv2_model.h5')
            new_image = load_image("/home/pi/saved_img-final.jpg")
            pred = new_model.predict(new_image)
            print(pred)
            #output=np.argmax(pred[0][0], axis=-1)
            output = pred[0][0]
            print(output)
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(11,GPIO.OUT)
            servo1 = GPIO.PWM(11,50) # pin 11 for servo1
            GPIO.setup(13,GPIO.OUT)
            servo2 = GPIO.PWM(13,50) # pin 13 for servo2
            
            # Start PWM running on both servos, value of 0 (pulse off)
            servo1.start(0)
            servo2.start(0)     
            #running the servos  
            if output > -0.5:   # plastic
                print("running plastic servo")
                servo1.ChangeDutyCycle(7)
                time.sleep(0.5)
                servo1.ChangeDutyCycle(0)
                time.sleep(3)
                servo1.ChangeDutyCycle(2)
                time.sleep(0.5)
                servo1.ChangeDutyCycle(0)
                time.sleep(0.5)
            
            else:
                print("running non-plastic servo")
                servo2.ChangeDutyCycle(7)
                time.sleep(0.5)
                servo2.ChangeDutyCycle(0)
                time.sleep(3)
                servo2.ChangeDutyCycle(2)
                time.sleep(0.5)
                servo2.ChangeDutyCycle(0)
                time.sleep(0.5)
            break
        
        
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            cameraOn = False
            break
    
    
servo1.stop()
servo2.stop()
GPIO.cleanup()


    
    
    




