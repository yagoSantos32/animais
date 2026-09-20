import cv2
from ultralytics import YOLO
import os

VIDEO = 'video32.mp4'
MODEL = 'yolov8n.pt'
DOG_CLASS_ID = 16

def main():
    if not os.path.exists(VIDEO):
        print(f"Vídeo não encontrado: {VIDEO}")
        return
    if not os.path.exists(MODEL):
        print(f"Modelo não encontrado: {MODEL}")
        return

    model = YOLO(MODEL)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return

    os.makedirs('diag_frames', exist_ok=True)
    frame_idx = 0
    dog_found = 0

    while frame_idx < 200:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, verbose=False)
        any_dog = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xy = [float(x) for x in box.xyxy[0]]
                if cls_id == DOG_CLASS_ID:
                    any_dog = True
                    dog_found += 1
                    print(f"Frame {frame_idx}: DOG detected conf={conf:.3f} xy={xy}")
        if any_dog:
            # save frame for inspection
            outp = f"diag_frames/frame_{frame_idx}.jpg"
            cv2.imwrite(outp, frame)
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames, dogs found so far: {dog_found}")

    print(f"Done. Processed {frame_idx} frames, total dog detections: {dog_found}")
    cap.release()

if __name__ == '__main__':
    main()
