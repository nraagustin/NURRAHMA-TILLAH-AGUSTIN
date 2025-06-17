import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from datetime import datetime
from collections import deque
import threading
import queue
from ultralytics import YOLO
import argparse

CATEGORIES = [ "DUDUK", "BERDIRI", "JATUH"]
FRAME_COUNT = 19
FRAME_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.7
ALERT_THRESHOLD = 0.5   
TARGET_FPS = 15  

COLOR_MAP = {
    "BERDIRI": (0, 255, 0),    
    "DUDUK": (255, 0, 0),      
    "JATUH": (0, 0, 255)     
}

class RealTimeActionClassifier:
    def __init__(self, model_path, camera_source=0, use_yolo=True):
        self.model_path = model_path
        self.camera_source = camera_source
        self.use_yolo = use_yolo
        
        self.frame_buffer = deque(maxlen=FRAME_COUNT)
        self.processed_buffer = deque(maxlen=FRAME_COUNT)
        
        self.current_prediction = "LOADING..."
        self.current_confidence = 0.0
        self.prediction_history = deque(maxlen=10) 
        
        self.is_running = False
        self.detection_active = False
        
        self.prediction_queue = queue.Queue(maxsize=2)
        self.prediction_thread = None
        
        self.fall_detected = False
        self.fall_alert_time = None
        self.alert_duration = 3.0  
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
 
        self.target_fps = TARGET_FPS
        self.frame_time = 1.0 / self.target_fps 
        self.last_frame_time = time.time()
        self.load_models()

    def load_models(self):
        print("Loading models...")
        try:
            self.classifier = load_model(self.model_path)
            print(f"Classification model loaded: {self.model_path}")
        except Exception as e:
            print(f"Error loading classification model: {e}")
            raise
        if self.use_yolo:
            try:
                self.detector = YOLO('yolo11n.pt')
                print(" YOLO model loaded")
            except Exception as e:
                print(f"Error loading YOLO: {e}")
                print("Fallback to HOG detector")
                self.use_yolo = False
                self.detector = cv2.HOGDescriptor()
                self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            self.detector = cv2.HOGDescriptor()
            self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("HOG detector loaded")
    
    def detect_humans(self, frame):
        try:
            if self.use_yolo:
                results = self.detector(frame, verbose=False)
                human_boxes = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])       
                            if class_id == 0 and confidence > 0.5: 
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                                human_boxes.append([x, y, w, h, confidence])
                
                return human_boxes
            else:
                boxes, weights = self.detector.detectMultiScale(frame, winStride=(8,8), 
                                                              padding=(32,32), scale=1.05)
                human_boxes = []
                for i, box in enumerate(boxes):
                    if weights[i] > 0.5:
                        x, y, w, h = box
                        human_boxes.append([x, y, w, h, weights[i]])
                
                return human_boxes
                
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def preprocess_frame(self, frame):
        try:
            processed = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            processed = processed.astype(np.float32) / 255.0
            return processed
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict_action(self):
        while self.is_running:
            try:
                if len(self.processed_buffer) == FRAME_COUNT:
                    frames_array = np.array(list(self.processed_buffer))
                    frames_input = np.expand_dims(frames_array, axis=0)
                    start_time = time.time()
                    predictions = self.classifier.predict(frames_input, verbose=0)
                    prediction_time = time.time() - start_time
                    predicted_class = np.argmax(predictions[0])
                    confidence = float(np.max(predictions[0]))
                    predicted_label = CATEGORIES[predicted_class]
                    self.current_prediction = predicted_label
                    self.current_confidence = confidence * 100
                    self.prediction_history.append({
                        'label': predicted_label,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
                    if predicted_label == "JATUH" and confidence > ALERT_THRESHOLD:
                        if not self.fall_detected:
                            self.fall_detected = True
                            self.fall_alert_time = time.time()
                            print(f"FALL DETECTED! Confidence: {confidence*100:.1f}%") 
                    if self.fall_detected and self.fall_alert_time:
                        if time.time() - self.fall_alert_time > self.alert_duration:
                            self.fall_detected = False
                            self.fall_alert_time = None
                time.sleep(self.frame_time)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                time.sleep(0.5)
    
    def draw_interface(self, frame):
        height, width = frame.shape[:2]
        

        overlay = frame.copy()
        panel_height = 140
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        

        cv2.putText(frame, f"Action: {self.current_prediction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {self.current_confidence:.1f}%", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f} (Target: {self.target_fps})", (10, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame Time: {self.frame_time*1000:.1f}ms", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        buffer_status = f"Buffer: {len(self.processed_buffer)}/{FRAME_COUNT}"
        cv2.putText(frame, buffer_status, (width-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        detection_color = (0, 255, 0) if self.detection_active else (0, 0, 255)
        status_text = "DETECTING" if self.detection_active else "NO HUMAN"
        cv2.putText(frame, status_text, (width-200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 1)
        

        if self.fall_detected:
            blink = int(time.time() * 4) % 2
            if blink:
                alert_overlay = frame.copy()
                cv2.rectangle(alert_overlay, (0, 0), (width, height), (0, 0, 255), 20)
                cv2.addWeighted(alert_overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "FALL DETECTED!", (width//2-150, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        

        if len(self.prediction_history) > 1:
            chart_x = width - 300
            chart_y = height - 100
            chart_w = 250
            chart_h = 80
            
            cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), 
                         (50, 50, 50), -1)
            cv2.putText(frame, "Prediction History", (chart_x + 5, chart_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            

            history_list = list(self.prediction_history)[-10:] 
            for i, pred in enumerate(history_list):
                color = COLOR_MAP.get(pred['label'], (255, 255, 255))
                bar_height = int(pred['confidence'] * 50) 
                bar_x = chart_x + 10 + i * 20
                bar_y = chart_y + chart_h - 10
                
                cv2.rectangle(frame, (bar_x, bar_y - bar_height), (bar_x + 15, bar_y), 
                             color, -1)
        
        return frame
    
    def draw_detections(self, frame, detections):
        for detection in detections:
            x, y, w, h = detection[:4]
            confidence = detection[4] if len(detection) > 4 else 1.0
            color = COLOR_MAP.get(self.current_prediction, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{self.current_prediction}: {self.current_confidence:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
    
    def update_fps(self):
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def run(self):
        print(f"Starting real-time action classification at {self.target_fps} FPS")
        print("Controls:")
        print("  SPACE: Toggle detection")
        print("  'r': Reset fall alert")
        print("  'q' or ESC: Quit")
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            print(f"Error: Cannot open camera source {self.camera_source}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)  
        
        self.is_running = True
        self.prediction_thread = threading.Thread(target=self.predict_action)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
        try:
            while self.is_running:
                current_time = time.time()
                time_since_last_frame = current_time - self.last_frame_time
                
                if time_since_last_frame < self.frame_time:
                    sleep_time = self.frame_time - time_since_last_frame
                    time.sleep(sleep_time)
                
                self.last_frame_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                frame = cv2.flip(frame, 1)
                detections = self.detect_humans(frame)
                self.detection_active = len(detections) > 0
                
                if self.detection_active:
                    processed_frame = self.preprocess_frame(frame)
                    if processed_frame is not None:
                        self.processed_buffer.append(processed_frame)
                    frame = self.draw_detections(frame, detections)
                else:
                    self.processed_buffer.clear()
                    self.current_prediction = "NO HUMAN"
                    self.current_confidence = 0.0
                frame = self.draw_interface(frame)
                
                self.update_fps()
                cv2.imshow('Real-Time Action Classification', frame)
                key = cv2.waitKey(int(1000/self.target_fps)) & 0xFF
                if key == ord('q') or key == 27:  
                    break
                elif key == ord(' '):  
                    pass  
                elif key == ord('r'): 
                    self.fall_detected = False
                    self.fall_alert_time = None
                    print("Fall alert reset")
                elif key == ord('s'):  
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"action_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            
            if self.prediction_thread and self.prediction_thread.is_alive():
                self.prediction_thread.join(timeout=2)
            
            print("Real-time classification stopped.")

def main():
    parser = argparse.ArgumentParser(description='Real-time Action Classification')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera source (default: 0 for webcam)')
    parser.add_argument('--video', type=str, default=None,
                       help='Video file path (alternative to camera)')
    parser.add_argument('--detector', choices=['yolo', 'hog'], default='yolo',
                       help='Human detector to use (default: yolo)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Target FPS (default: 5)')
    
    args = parser.parse_args()
    
    camera_source = args.video if args.video else args.camera
    use_yolo = args.detector == 'yolo'
    
    global TARGET_FPS
    TARGET_FPS = args.fps
    
    classifier = RealTimeActionClassifier(
        model_path=args.model,
        camera_source=camera_source,
        use_yolo=use_yolo
    )
    classifier.target_fps = TARGET_FPS
    classifier.frame_time = 1.0 / TARGET_FPS
    
    classifier.run()

if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\Lenovo\OneDrive\Documents\TA\TA TESTING\3model9.h5"

    TARGET_FPS = 5 
    classifier = RealTimeActionClassifier(
        model_path=MODEL_PATH,
        camera_source=0,  
        use_yolo=True
    )
    classifier.target_fps = TARGET_FPS
    classifier.frame_time = 1.0 / TARGET_FPS
    
    classifier.run()
