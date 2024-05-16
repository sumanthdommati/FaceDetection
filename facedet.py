import cv2
import numpy as np
import face_recognition as fr
video_capture = cv2.VideoCapture(0)
image = fr.load_image_file("D:/WhatsApp Image 2024-05-15 at 16.37.11_be41789a.jpg")
image_face_encoding = fr.face_encodings(image)[0]
known_face_encodings = [image_face_encoding]
known_face_names = ["Sumanth"]
while True:
    ret, frame = video_capture.read()
    rframe=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    fc_locations = fr.face_locations(rframe)
    fc_encodings=fr.face_encodings(rframe, fc_locations)
    for(top, right, bottom, left), face_encoding in zip(fc_locations, fc_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "unknown"
        fc_distances = fr.face_distance(known_face_encodings, face_encoding)
        match_index = np.argmin(fc_distances)
        if matches[match_index]:
            name = known_face_names[match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left +6, bottom- 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow("Face Detection Using Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
