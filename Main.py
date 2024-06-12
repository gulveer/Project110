# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("keras_modal.h5")
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    img = cv2.resize(frame, (224,224))
    testimage = np.array(img, dtype = np.float32)
    testimage = np.expand_dims(testimage, axis = 0)
    normaliseimage = testimage/255
    prediction = model.predict(normaliseimage)
    print("PREDICTION: ", prediction)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()