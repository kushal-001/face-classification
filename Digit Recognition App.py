# %%  Import libraries

import numpy as np
from sklearn.externals import joblib
import cv2 as cv
# %%  Load model

identifier = joblib.load('G:\Machine Learning\Project\Digit Recognition/Digit_Recnogition_App')

# %%  Webcam

webcam = cv.VideoCapture(0)
webcam.set(3, 450)
webcam.set(4, 800)

# %%  Loop through the frames

while webcam.isOpened():

    status, frame = webcam.read()

    image = cv.resize(frame, (8,8))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image.reshape(64,)
    label = identifier.predict([image])

    cv.putText(frame, 'Digit: ', (20,35), cv.FONT_HERSHEY_TRIPLEX, 1.6, (0,0,0), 2)
    cv.putText(frame, str(label), (180,35), cv.FONT_HERSHEY_TRIPLEX, 1.6, (0,0,0), 2)
    cv.imshow('Digit Recognition', frame)

    if cv.waitKey(1) & 0xff == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()





# %%
