import cv2

# Load the cascade
p='C:\Python38\Scripts\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(p)

# To capture video from webcam.
cam = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:

    # Read the frame
    _, img = cam.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', gray)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cam.release()
cv2.destroyAllWindows()

"""
my project is face detection using har cascade classifier"
in that we are frontalface.xml file .it has frontal face features.
the featires can be extracted throuh our images if our image is gray color.
if can detecet the face and draw the rectangle on it:
"""

