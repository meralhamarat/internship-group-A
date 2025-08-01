import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

mp_pose = mp.solutions.pose

# --- AYARLANACAK YOLLAR ---
# Video dosyalarınızın bulunduğu ANA KLASÖR.
# Örneğin: 'C:/Users/Meral PC/Videos/MyActionVideos/' gibi bir yol olmalı.
# Bu klasörün içinde "Clapping", "Sitting", "Standing Still" gibi alt klasörleriniz olmalı.
video_root_dir = 'C:/Users/Meral PC/OneDrive/Masaüstü/pose_classification/dataset/' 
# Keypoint CSV'lerinin kaydedileceği dizin.
output_csv_dir = 'C:/Users/Meral PC/OneDrive/Masaüstü/pose_classification/dataset/'
# --- AYARLAMALAR BİTTİ ---

# Her X. kareyi işle. Bu, CSV'lerin boyutunu ve modelin işleyeceği dizi uzunluğunu azaltır.
FRAME_SKIP_RATE = 5 # Her 5. kareyi işleyecek.

print("--- Keypoint Çıkarma Başlatılıyor ---")

os.makedirs(output_csv_dir, exist_ok=True)

for action_folder in os.listdir(video_root_dir):
    action_video_path = os.path.join(video_root_dir, action_folder)
    output_action_csv_path = os.path.join(output_csv_dir, action_folder)
    
    if os.path.isdir(action_video_path):
        os.makedirs(output_action_csv_path, exist_ok=True)
        print(f"\nİşleniyor: {action_folder} klasörü")
        
        for video_file in os.listdir(action_video_path):
            video_full_path = os.path.join(action_video_path, video_file)
            
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  Video: {video_file} işleniyor...")
                
                cap = cv2.VideoCapture(video_full_path)
                if not cap.isOpened():
                    print(f"    Hata: '{video_file}' açılamadı. Atlanıyor.")
                    continue
                
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    video_keypoints_list = [] 
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_count % FRAME_SKIP_RATE == 0:
                            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(image_rgb)
                            
                            if results.pose_landmarks:
                                frame_keypoints = []
                                for landmark in results.pose_landmarks.landmark:
                                    frame_keypoints.extend([landmark.x, landmark.y])
                                video_keypoints_list.append(frame_keypoints)
                        
                        frame_count += 1
                            
                    cap.release()
                    
                    if video_keypoints_list:
                        csv_output_filename = os.path.splitext(video_file)[0] + '.csv'
                        csv_output_full_path = os.path.join(output_action_csv_path, csv_output_filename)
                        
                        df_keypoints = pd.DataFrame(video_keypoints_list)
                        df_keypoints.to_csv(csv_output_full_path, index=False)
                        print(f"    Keypointler kaydedildi: {csv_output_full_path}")
                    else:
                        print(f"    Uyarı: '{video_file}' videosundan keypoint çıkarılamadı veya boş. Atlanıyor.")

print("\n--- Tüm keypoint çıkarma ve CSV kaydetme tamamlandı! ---")