import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib
from collections import deque
from mediapipe.framework.formats import landmark_pb2

# --- MediaPipe ve YOLO Modelleri ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

yolo_model = YOLO('yolov8n.pt') 

# --- Parametreler ve Sabitler ---
VIDEO_SOURCE = 'man.mp4'
SEQUENCE_LENGTH = 10
MIN_CONFIDENCE_THRESHOLD = 0.3
MIN_YOLO_CONFIDENCE = 0.5
PERSON_CLASS_ID = 0
FRAME_SKIP_RATE = 2 
MAX_INVISIBLE_TIME = 1.0  # Tracker'ın kaybolması için geçen süre (saniye)

def load_trained_model():
    """Eğitilmiş modeli ve label encoder'ı yükle"""
    try:
        model = load_model('best_action_model.h5')
        label_encoder = joblib.load('label_encoder.pkl')
        print("Hareket tanıma modeli ve label encoder başarıyla yüklendi!")
        print(f"Sınıflar: {label_encoder.classes_}")
        return model, label_encoder
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return None, None

def setup_log_file():
    """Log dosyasını hazırlar"""
    try:
        with open('person_actions_log.csv', 'w', newline='') as csvfile:
            fieldnames = ['person_id', 'action', 'duration_seconds', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print("Log dosyası hazırlandı: person_actions_log.csv")
    except Exception as e:
        print(f"Log dosyası oluşturulurken hata: {e}")

def append_to_csv_log(person_id, action, duration):
    """CSV dosyasına yeni bir log satırı ekler"""
    try:
        with open('person_actions_log.csv', 'a', newline='') as csvfile:
            fieldnames = ['person_id', 'action', 'duration_seconds', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'person_id': person_id,
                'action': action,
                'duration_seconds': f"{duration:.2f}",
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
    except Exception as e:
        print(f"Log dosyasına yazarken hata: {e}")

def extract_pose_keypoints(pose_landmarks):
    """Pose landmark'larından keypoint'leri çıkar"""
    if pose_landmarks and pose_landmarks.landmark:
        keypoints = []
        for landmark in pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y])
        return keypoints
    return None

def predict_action(model, label_encoder, keypoints_sequence):
    """Keypoint dizisinden hareket tahmini yap"""
    if len(keypoints_sequence) < SEQUENCE_LENGTH:
        return "Insufficient Data", 0.0
    
    recent_sequence = keypoints_sequence[-SEQUENCE_LENGTH:]
    sequence_array = np.array(recent_sequence).reshape(1, SEQUENCE_LENGTH, -1)
    
    predictions = model.predict(sequence_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    predicted_action = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_action, confidence

class PersonTracker:
    """Tek bir kişiye ait takip bilgilerini tutar"""
    def __init__(self, track_id, initial_pose_landmarks, initial_frame_time):
        self.id = track_id
        self.keypoints_history = deque(maxlen=SEQUENCE_LENGTH * 2)
        self.last_predicted_action = "Unknown"
        self.last_action_start_time = initial_frame_time
        self.current_action_duration = 0.0
        self.last_pose_landmarks = initial_pose_landmarks
        self.last_update_time = initial_frame_time
        
    def update(self, new_pose_landmarks, frame_time, model, label_encoder):
        """Tracker'ı yeni poz bilgisiyle güncelle ve hareket tahmini yap"""
        keypoints = extract_pose_keypoints(new_pose_landmarks)
        if keypoints:
            self.keypoints_history.append(keypoints)
        
        if len(self.keypoints_history) >= SEQUENCE_LENGTH:
            predicted_action, confidence = predict_action(model, label_encoder, list(self.keypoints_history))
            
            if confidence > MIN_CONFIDENCE_THRESHOLD:
                if predicted_action != self.last_predicted_action:
                    # Yeni bir eylem başladıysa, eskisini logla
                    if self.last_predicted_action != "Unknown":
                        duration = frame_time - self.last_action_start_time
                        if duration > 0.1: # Çok kısa eylemleri loglama
                            append_to_csv_log(self.id, self.last_predicted_action, duration)
                    
                    self.last_predicted_action = predicted_action
                    self.last_action_start_time = frame_time
                self.current_action_duration = frame_time - self.last_action_start_time
            else:
                self.last_predicted_action = "Unknown"
                self.current_action_duration = 0.0
            
        self.last_pose_landmarks = new_pose_landmarks
        self.last_update_time = frame_time
        return self.last_predicted_action, self.current_action_duration

def main():
    model, label_encoder = load_trained_model()
    if model is None:
        print("Model yüklenemedi. Program sonlandırılıyor.")
        return

    setup_log_file()
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Video kaynağı açılamadı: {VIDEO_SOURCE}")
        return

    print("YOLO, ByteTrack ve Hareket Tanıma entegrasyonu başlatıldı...")
    print("Çıkış için ESC tuşuna basın")
    
    person_trackers = {}
    frame_count = 0
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Video bittiğinde tüm aktif tracker'ları logla
                for tracker_id, tracker in person_trackers.items():
                    if tracker.last_predicted_action != "Unknown":
                        append_to_csv_log(tracker_id, tracker.last_predicted_action, tracker.current_action_duration)
                break
            
            h, w, _ = frame.shape
            frame_count += 1
            frame_time = time.time()
            
            # Kare atlama mantığı ile performansı artırma
            if frame_count % FRAME_SKIP_RATE == 0:
                yolo_results = yolo_model.track(frame, persist=True, classes=PERSON_CLASS_ID, conf=MIN_YOLO_CONFIDENCE, verbose=False)
                
                detected_ids_this_frame = set()

                if yolo_results and yolo_results[0].boxes.id is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, ids):
                        detected_ids_this_frame.add(track_id)
                        x1, y1, x2, y2 = box
                        
                        padding = 10 
                        x1_pad = max(0, x1 - padding)
                        y1_pad = max(0, y1 - padding)
                        x2_pad = min(w, x2 + padding)
                        y2_pad = min(h, y2 + padding)
                        
                        cropped_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                        
                        if cropped_img.size == 0:
                            continue
                            
                        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(cropped_img_rgb)
                        
                        if pose_results.pose_landmarks:
                            adjusted_landmarks = pose_results.pose_landmarks
                            for landmark in adjusted_landmarks.landmark:
                                landmark.x = (landmark.x * (x2_pad - x1_pad) + x1_pad) / w
                                landmark.y = (landmark.y * (y2_pad - y1_pad) + y1_pad) / h
                                
                            if track_id not in person_trackers:
                                person_trackers[track_id] = PersonTracker(track_id, adjusted_landmarks, frame_time)
                            else:
                                person_trackers[track_id].update(adjusted_landmarks, frame_time, model, label_encoder)

            # --- Tracker Yönetimi ve Çizimler ---
            y_offset = 30
            trackers_to_keep = {}
            for tracker_id, tracker in person_trackers.items():
                if (frame_time - tracker.last_update_time) < MAX_INVISIBLE_TIME:
                    trackers_to_keep[tracker_id] = tracker
                    
                    # bounding box ve metinleri çiz
                    x_coords = [lm.x * w for lm in tracker.last_pose_landmarks.landmark]
                    y_coords = [lm.y * h for lm in tracker.last_pose_landmarks.landmark]
                    if x_coords and y_coords:
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        text = f'ID: {tracker_id}'
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        mp_drawing.draw_landmarks(
                            frame,
                            tracker.last_pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # Ekranın kenarına eylem bilgilerini yaz
                    action_text = f"ID:{tracker_id} | {tracker.last_predicted_action} | {tracker.current_action_duration:.1f}s"
                    cv2.putText(frame, action_text, (w - 300, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    y_offset += 30
                else:
                    # Bir tracker kalıcı olarak kaybolduysa son eylemini logla
                    if tracker.last_predicted_action != "Unknown":
                         append_to_csv_log(tracker_id, tracker.last_predicted_action, tracker.current_action_duration)
            
            person_trackers = trackers_to_keep
            
            cv2.imshow('YOLO + ByteTrack + MediaPipe + Action Recognition', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                # Çıkışta tüm aktif tracker'ları logla
                for tracker_id, tracker in person_trackers.items():
                    if tracker.last_predicted_action != "Unknown":
                        append_to_csv_log(tracker_id, tracker.last_predicted_action, tracker.current_action_duration)
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program sonlandırıldı.")

if __name__ == "__main__":
    main()