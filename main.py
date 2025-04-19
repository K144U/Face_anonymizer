import os
import cv2
import mediapipe as mp
import argparse


def process_img(img, face_detection):
    H, W, _ = img.shape  # Calculate inside the function
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (50, 50))

    return img


args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default= None)
args = args.parse_args()


mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:

    if args.mode == 'image':
        img = cv2.imread(args.filePath)

        if img is None:
            print(f"Image not found at {args.filePath}")
            exit()

        img = process_img(img, face_detection)

        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter('output.mp4',
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       15, (int(cap.get(3)), int(cap.get(4))))

        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    elif args.mode == 'webcam':
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_img(frame, face_detection)

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
