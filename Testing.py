import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import time
from datetime import datetime 
from ultralytics import YOLO
import random
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
 
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '0'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
  
BASE_DATASET_PATH = r"C:\Users\Lenovo\OneDrive\Documents\TA\TESTING RESULT"
MODEL_PATH = r"C:\Users\Lenovo\OneDrive\Documents\TA\TA TESTING\3model1.h5"
CATEGORIES = ["BERDIRI", "DUDUK", "JATUH"]
FRAME_COUNT = 20
FRAME_SIZE = (64, 64)
TARGET_DURATION = 2.0  
MIN_VIDEO_DURATION = 1.0  
COLOR_MAP = {
    "BERDIRI": (0, 255, 0),    
    "DUDUK": (255, 0, 0),    
    "JATUH": (0, 0, 255)      
}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = r"C:\Users\Lenovo\OneDrive\Documents\TA\TESTING RESULT"
OUTPUT_DIR = os.path.join(BASE_DIR, f"testing_results_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output akan disimpan di: {OUTPUT_DIR}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Available devices: {tf.config.list_physical_devices()}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
def configure_tensorflow():
    tf.config.experimental.enable_op_determinism()  
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}") 
    tf.keras.backend.set_floatx('float32')
    print(f"TensorFlow configured - Float type: {tf.keras.backend.floatx()}")
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0 
    cap.release()  
    video_info = {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
        'valid': duration >= MIN_VIDEO_DURATION and total_frames > 0
    }
    return video_info

def analyze_motion_intensity(video_path):
    cap = cv2.VideoCapture(video_path) 
    motion_scores = []
    prev_frame = None
    frame_idx = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
       
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            motion_magnitude = np.mean(diff)
            motion_scores.append(motion_magnitude)
        else:
            motion_scores.append(0)  
        prev_frame = gray
        frame_idx += 1
    cap.release()
    return np.array(motion_scores)

def detect_action_phases(motion_scores, fps):
    if len(motion_scores) == 0:
        return [], motion_scores, []
    window_size = max(3, int(fps * 0.2))  
    if window_size >= len(motion_scores):
        window_size = max(1, len(motion_scores) // 3)
    smoothed_scores = np.convolve(motion_scores, np.ones(window_size)/window_size, mode='same')
    if len(smoothed_scores) > 0:
        motion_threshold = np.percentile(smoothed_scores, 75)  
        high_motion_frames = smoothed_scores > motion_threshold
        try:
            min_distance = max(1, int(fps * 0.5))
            peaks, _ = find_peaks(smoothed_scores, height=motion_threshold, distance=min_distance)
        except:
            peaks = []
    else:
        peaks = []
    phases = []
    total_frames = len(motion_scores)
    if len(peaks) > 0:
        main_action_start = max(0, peaks[0] - int(fps * 0.5)) 
        main_action_end = min(total_frames, peaks[-1] + int(fps * 0.5)) 
        phases = [
            {
                'name': 'pre_action',
                'start': 0,
                'end': main_action_start,
                'importance': 0.2  
            },
            {
                'name': 'main_action',
                'start': main_action_start,
                'end': main_action_end,
                'importance': 0.6 
            },
            {
                'name': 'post_action',
                'start': main_action_end,
                'end': total_frames,
                'importance': 0.2
            }
        ]
    else:
        phases = [{
            'name': 'uniform',
            'start': 0,
            'end': total_frames,
            'importance': 1.0
        }]
    return phases, smoothed_scores, peaks

def sample_frames_action_aware(video_path, target_frames=FRAME_COUNT, target_duration=TARGET_DURATION):
    video_info = get_video_info(video_path)
    if not video_info or not video_info['valid']:
        return None, None, None
    cap = cv2.VideoCapture(video_path)
    total_frames = video_info['total_frames']
    fps = video_info['fps']
    duration = video_info['duration']
    try:
        motion_scores = analyze_motion_intensity(video_path)
        phases, smoothed_motion, peaks = detect_action_phases(motion_scores, fps)
    except Exception as e:
        print(f"Motion analysis failed for {video_path}: {e}")
        return sample_frames_by_duration(video_path, target_frames, target_duration)

    if duration <= target_duration:
        start_frame, end_frame = 0, total_frames - 1
        sampling_strategy = "full_video_action_aware"
    else:
        if len(peaks) > 0 and len(smoothed_motion) > 0:
            action_center = peaks[np.argmax(smoothed_motion[peaks])]
            half_duration_frames = int((target_duration * fps) / 2)
            start_frame = max(0, action_center - half_duration_frames)
            end_frame = min(total_frames - 1, action_center + half_duration_frames)
            sampling_strategy = "action_centered"
        else:
            start_time = (duration - target_duration) / 2
            start_frame = int(start_time * fps)
            end_frame = int((start_time + target_duration) * fps)
            sampling_strategy = "middle_fallback"
    frame_indices = []
    for phase in phases:
        phase_start = max(start_frame, phase['start'])
        phase_end = min(end_frame, phase['end'])
       
        if phase_start >= phase_end:
            continue
        phase_frames = int(target_frames * phase['importance'])
       
        if phase_frames > 0:
            if phase['name'] == 'main_action':
                phase_indices = np.linspace(phase_start, phase_end,
                                          min(phase_frames, phase_end - phase_start + 1),
                                          dtype=int)
            else:
                available_frames = phase_end - phase_start + 1
                if available_frames <= phase_frames:
                    phase_indices = list(range(phase_start, phase_end + 1))
                else:
                    phase_indices = np.linspace(phase_start, phase_end, phase_frames, dtype=int)
           
            frame_indices.extend(phase_indices)
    frame_indices = sorted(list(set(frame_indices)))
    if len(frame_indices) > target_frames:
        action_frames = []
        other_frames = []
       
        for idx in frame_indices:
            if any(phase['name'] == 'main_action' and
                   phase['start'] <= idx <= phase['end'] for phase in phases):
                action_frames.append(idx)
            else:
                other_frames.append(idx)
        if len(action_frames) <= target_frames:
            remaining_slots = target_frames - len(action_frames)
            if remaining_slots > 0 and other_frames:
                step = max(1, len(other_frames) // remaining_slots)
                selected_others = other_frames[::step][:remaining_slots]
                frame_indices = sorted(action_frames + selected_others)
            else:
                frame_indices = action_frames
        else:
            step = max(1, len(action_frames) // target_frames)
            frame_indices = sorted(action_frames[::step][:target_frames])


    elif len(frame_indices) < target_frames:
        all_available = list(range(start_frame, end_frame + 1))
        additional_needed = target_frames - len(frame_indices)
        candidates = [f for f in all_available if f not in frame_indices]
        if candidates:
            step = max(1, len(candidates) // additional_needed) if additional_needed > 0 else 1
            additional_frames = candidates[::step][:additional_needed]
            frame_indices.extend(additional_frames)
            frame_indices = sorted(list(set(frame_indices)))
    frames = []
    original_frames = []
    successful_extractions = 0
    for idx in frame_indices[:target_frames]:  
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
       
        if ret and frame is not None:
            original_frames.append(frame.copy())
           
            processed_frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            processed_frame = processed_frame.astype(np.float32) / 255.0
            frames.append(processed_frame)
            successful_extractions += 1
    cap.release()
    while len(frames) < target_frames:
        if frames:
            frames.append(frames[-1].copy())
            if original_frames:
                original_frames.append(original_frames[-1].copy())
        else:
            black_frame = np.zeros((*FRAME_SIZE, 3), dtype=np.float32)
            frames.append(black_frame)
            original_frames.append(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))
    sampling_info = {
        'video_path': video_path,
        'original_duration': duration,
        'original_fps': fps,
        'sampling_strategy': sampling_strategy,
        'motion_analysis': {
            'motion_scores': motion_scores.tolist() if len(motion_scores) > 0 else [],
            'peaks': peaks.tolist() if len(peaks) > 0 else [],
            'phases': phases
        },
        'frame_indices': frame_indices[:target_frames],
        'successful_extractions': successful_extractions,
        'final_frame_count': len(frames)
    }
    return np.array(frames, dtype=np.float32), original_frames, sampling_info

def sample_frames_by_duration(video_path, target_frames=FRAME_COUNT, target_duration=TARGET_DURATION):
    video_info = get_video_info(video_path)
    if not video_info or not video_info['valid']:
        print(f"Invalid video: {video_path}")
        return None, None, None
    cap = cv2.VideoCapture(video_path)
    total_frames = video_info['total_frames']
    fps = video_info['fps']
    duration = video_info['duration']
   
    if duration <= target_duration:
        sampling_strategy = "full_video"
        start_time = 0
        end_time = duration
        sample_duration = duration
    else:
        sampling_strategy = "duration_based"
        start_time = (duration - target_duration) / 2
        end_time = start_time + target_duration
        sample_duration = target_duration
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)
    available_frames = end_frame - start_frame + 1
    if available_frames >= target_frames:
        frame_indices = np.linspace(start_frame, end_frame, target_frames, dtype=int)
    else:
        frame_indices = list(range(start_frame, end_frame + 1))
    frames = []
    original_frames = []
    successful_extractions = 0
   
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
       
        if ret and frame is not None:
            original_frames.append(frame.copy())
            processed_frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            processed_frame = processed_frame.astype(np.float32)
            processed_frame = processed_frame / 255.0
            frames.append(processed_frame)
            successful_extractions += 1
        else:
            print(f"Warning: Could not read frame {idx} from {video_path}")
    cap.release()
    while len(frames) < target_frames:
        if frames:
            frames.append(frames[-1].copy())
            if original_frames:
                original_frames.append(original_frames[-1].copy())
        else:
            black_frame = np.zeros((*FRAME_SIZE, 3), dtype=np.float32)
            frames.append(black_frame)
            black_original = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            original_frames.append(black_original)
    frames = frames[:target_frames]
    original_frames = original_frames[:target_frames]
    sampling_info = {
        'video_path': video_path,
        'original_duration': duration,
        'original_fps': fps,
        'original_total_frames': total_frames,
        'sampling_strategy': sampling_strategy,
        'sample_start_time': start_time,
        'sample_end_time': end_time,
        'sample_duration': sample_duration,
        'frame_indices': frame_indices.tolist() if isinstance(frame_indices, np.ndarray) else frame_indices,
        'successful_extractions': successful_extractions,
        'padding_applied': target_frames - successful_extractions,
        'final_frame_count': len(frames)
    }
    return np.array(frames, dtype=np.float32), original_frames, sampling_info
def initialize_yolo():
    try:
        model = YOLO('yolo11n.pt')  
        print("YOLOv11 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLOv11: {e}")
        print("Make sure you have the latest ultralytics version: pip install ultralytics>=8.3.0")
        return None
def initialize_hog_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def detect_humans_yolo(frame, yolo_model):
    try:
        results = yolo_model(frame, verbose=False)
       
        human_boxes = []
       
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])   
                    if class_id == 0 and confidence > 0.5:  
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)
                        human_boxes.append([x, y, w, h])
        return human_boxes  
    except Exception as e:
        print(f"Error in YOLOv11 detection: {e}")
        return []


def detect_humans_hog(frame, hog):
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
 
    human_boxes = []
    for i, box in enumerate(boxes):
        if weights[i] > 0.5:
            x, y, w, h = box
            human_boxes.append([x, y, w, h])
   
    return human_boxes

def load_dataset_action_aware(split):
    X, y, video_paths, sampling_info_list = [], [], [], []
    split_path = os.path.join(BASE_DATASET_PATH, split)  
   
    if not os.path.exists(split_path):
        print(f"Dataset path not found: {split_path}")
        return np.array([]), np.array([]), [], []
   
    print(f"Loading from: {split_path}")
    total_videos_processed = 0
    total_videos_skipped = 0
    for category in sorted(CATEGORIES):
        category_path = os.path.join(split_path, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category_path} tidak ditemukan.")
            continue  
        video_count = 0
        videos_skipped = 0
        video_files = sorted([f for f in os.listdir(category_path)
                             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])      
        print(f"\nProcessing category: {category}")
        print(f"Found {len(video_files)} video files")
       
        for i, video_name in enumerate(video_files):
            video_path = os.path.join(category_path, video_name)
            try:
                video_frames, original_frames, sampling_info = sample_frames_action_aware(video_path)
               
                if video_frames is not None and video_frames.shape[0] == FRAME_COUNT:
                    X.append(video_frames)
                    y.append(category)
                    video_paths.append(video_path)
                    sampling_info_list.append(sampling_info)
                    video_count += 1
                    total_videos_processed += 1
                   
                    if video_count <= 3:
                        print(f"  Video {video_count}: {video_name}")
                        print(f"    Duration: {sampling_info['original_duration']:.2f}s, "
                              f"FPS: {sampling_info['original_fps']:.1f}")
                        print(f"    Strategy: {sampling_info['sampling_strategy']}")
                        print(f"    Peaks: {len(sampling_info['motion_analysis']['peaks'])} detected")
                        print(f"    Phases: {[p['name'] for p in sampling_info['motion_analysis']['phases']]}")
                        print(f"    Extracted: {sampling_info['successful_extractions']}/{FRAME_COUNT} frames")
                else:
                    videos_skipped += 1
                    total_videos_skipped += 1
                    if videos_skipped <= 3:  
                        print(f"  Skipped: {video_name} (insufficient frames or invalid)")
                       
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                videos_skipped += 1
                total_videos_skipped += 1
                continue   
        print(f"Category {category}: Loaded {video_count} videos, Skipped {videos_skipped} videos")
   
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
   
    print(f"\nDataset loading summary:")
    print(f"Total videos processed: {total_videos_processed}")
    print(f"Total videos skipped: {total_videos_skipped}")
    print(f"Final dataset shape: {X.shape}")
   
    return X, y, video_paths, sampling_info_list


def visualize_motion_analysis(sampling_info, output_path):
    try:
        motion_scores = np.array(sampling_info['motion_analysis']['motion_scores'])
        peaks = sampling_info['motion_analysis']['peaks']
        phases = sampling_info['motion_analysis']['phases']
        frame_indices = sampling_info['frame_indices']
       
        if len(motion_scores) == 0:
            return
       
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(motion_scores, label='Motion Intensity', alpha=0.7)
        if len(peaks) > 0:
            plt.plot(peaks, motion_scores[peaks], 'ro', markersize=8, label='Action Peaks')
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for i, phase in enumerate(phases):
            plt.axvspan(phase['start'], phase['end'],
                       alpha=0.3, color=colors[i % len(colors)],
                       label=f"{phase['name']} (imp: {phase['importance']})")
        plt.xlabel('Frame Index')
        plt.ylabel('Motion Intensity')
        plt.title('Motion Analysis and Action Phases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        plt.scatter(frame_indices, [1]*len(frame_indices),
                   c='red', s=50, label='Selected Frames')
        plt.plot(motion_scores, alpha=0.5, label='Motion Intensity')
       
        for i, phase in enumerate(phases):
            plt.axvspan(phase['start'], phase['end'],
                       alpha=0.2, color=colors[i % len(colors)])
       
        plt.xlabel('Frame Index')
        plt.ylabel('Selection + Motion')
        plt.title('Frame Selection Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
       
    except Exception as e:
        print(f"Error creating motion analysis visualization: {e}")


def create_classified_video(original_frames, predicted_class, confidence, output_path, yolo_model=None, hog_detector=None):
    if not original_frames:
        return  
    height, width = original_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
   
    color = COLOR_MAP[predicted_class]
    label = f"{predicted_class}: {confidence:.2f}%"
   
    for frame_idx, frame in enumerate(original_frames):
        try:
            human_boxes = []
           
            if yolo_model is not None:
                human_boxes = detect_humans_yolo(frame, yolo_model)
            elif hog_detector is not None:
                human_boxes = detect_humans_hog(frame, hog_detector)
           
            if human_boxes:
                for box in human_boxes:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                   
                    label_y = max(y - 10, 20)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                   
                    cv2.rectangle(frame, (x, label_y - text_size[1] - 5),
                                (x + text_size[0] + 10, label_y + 5), (0, 0, 0), -1)
                   
                    cv2.putText(frame, label, (x + 5, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (20, 20), (width-20, height-20), color, 2)
                cv2.putText(frame, f"{label} (No Human Detected)", (30, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            out.write(frame)     
        except Exception as e:
            print(f"Detection failed for frame {frame_idx}: {e}")
            out.write(frame)
    out.release()

def save_sampling_analysis(sampling_info_list, output_dir):
    analysis_path = os.path.join(output_dir, "action_aware_sampling_analysis.txt")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ACTION-AWARE SAMPLING ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"Target Frames: {FRAME_COUNT}\n")
        f.write(f"Target Duration: {TARGET_DURATION}s\n")
        f.write(f"Minimum Video Duration: {MIN_VIDEO_DURATION}s\n")
        f.write(f"Frame Size: {FRAME_SIZE}\n\n")

        durations = [info['original_duration'] for info in sampling_info_list]
        fps_values = [info['original_fps'] for info in sampling_info_list]
        strategies = [info['sampling_strategy'] for info in sampling_info_list] 
        f.write("Dataset Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Videos: {len(sampling_info_list)}\n")
        f.write(f"Duration - Min: {min(durations):.2f}s, Max: {max(durations):.2f}s, Mean: {np.mean(durations):.2f}s\n")
        f.write(f"FPS - Min: {min(fps_values):.1f}, Max: {max(fps_values):.1f}, Mean: {np.mean(fps_values):.1f}\n\n")   
        strategy_counts = Counter(strategies)
        f.write("Sampling Strategies Used:\n")
        f.write("-" * 30 + "\n")
        for strategy, count in strategy_counts.items():
            percentage = (count / len(strategies)) * 100
            f.write(f"{strategy}: {count} videos ({percentage:.1f}%)\n")
        f.write("\n")
        f.write("Detailed Video Analysis:\n")
        f.write("-" * 50 + "\n")
       
        for i, info in enumerate(sampling_info_list):
            f.write(f"Video {i+1}: {os.path.basename(info['video_path'])}\n")
            f.write(f"  Duration: {info['original_duration']:.2f}s, FPS: {info['original_fps']:.1f}\n")
            f.write(f"  Strategy: {info['sampling_strategy']}\n")
            f.write(f"  Motion Peaks: {len(info['motion_analysis']['peaks'])}\n")
            f.write(f"  Phases: {len(info['motion_analysis']['phases'])}\n")
            f.write(f"  Frames Extracted: {info['successful_extractions']}/{FRAME_COUNT}\n")
            if i < 20:
                f.write(f"  Frame Indices: {info['frame_indices'][:10]}{'...' if len(info['frame_indices']) > 10 else ''}\n")
            f.write("\n")

def evaluate_model_with_timing():
    """Evaluate model with detailed timing analysis"""
    configure_tensorflow()
    
    print("Loading TensorFlow model...")
    model_load_start = time.time()
    model = load_model(MODEL_PATH)
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.4f} seconds")
    
    print("Initializing human detection models...")
    yolo_init_start = time.time()
    yolo_model = initialize_yolo()
    yolo_init_time = time.time() - yolo_init_start
    
    hog_init_start = time.time()
    hog_detector = initialize_hog_detector()
    hog_init_time = time.time() - hog_init_start
    
    print(f"YOLO initialization: {yolo_init_time:.4f} seconds")
    print(f"HOG initialization: {hog_init_time:.4f} seconds")
    
    print("Loading test dataset...")
    dataset_load_start = time.time()
    X_test, y_test, video_paths, sampling_info_list = load_dataset_action_aware('test')
    dataset_load_time = time.time() - dataset_load_start
    print(f"Dataset loaded in {dataset_load_time:.4f} seconds")
    
    if len(X_test) == 0:
        print("No test data found!")
        return None, None, None
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    print(f"\nTest dataset: {X_test.shape}, Labels: {len(set(y_test))}")
    print(f"Categories: {list(label_encoder.classes_)}")
    
    inference_times = []
    preprocessing_times = []
    postprocessing_times = []
    total_sample_times = []
    detection_times_yolo = []
    detection_times_hog = []
    predictions = []
    prediction_probabilities = []
    
    print("\nPerforming inference with timing measurement...")
    print("=" * 80)
    
    total_inference_start = time.time()
    
    for i in range(len(X_test)):
        sample_start_time = time.time()  
        preprocessing_start = time.time()
        X_sample = X_test[i:i+1]  
        preprocessing_time = time.time() - preprocessing_start
        preprocessing_times.append(preprocessing_time) 
        inference_start = time.time()
        prediction_probs = model.predict(X_sample, verbose=0)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        postprocessing_start = time.time()
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class = label_encoder.classes_[predicted_class_idx]
        confidence = prediction_probs[0][predicted_class_idx] * 100
        postprocessing_time = time.time() - postprocessing_start
        postprocessing_times.append(postprocessing_time)  
        predictions.append(predicted_class)
        prediction_probabilities.append(prediction_probs[0])     
        sample_total_time = time.time() - sample_start_time
        total_sample_times.append(sample_total_time)
        
        if len(sampling_info_list) > i:
            try:
                video_path = video_paths[i]
                cap = cv2.VideoCapture(video_path)
                ret, first_frame = cap.read()
                cap.release()
                if ret and first_frame is not None:
                    if yolo_model is not None:
                        yolo_detect_start = time.time()
                        yolo_boxes = detect_humans_yolo(first_frame, yolo_model)
                        yolo_detect_time = time.time() - yolo_detect_start
                        detection_times_yolo.append(yolo_detect_time)      
                    hog_detect_start = time.time()
                    hog_boxes = detect_humans_hog(first_frame, hog_detector)
                    hog_detect_time = time.time() - hog_detect_start
                    detection_times_hog.append(hog_detect_time)       
            except Exception as e:
                print(f"Detection timing failed for sample {i}: {e}") 
        if (i + 1) % 10 == 0 or (i + 1) == len(X_test):
            current_avg_inference = np.mean(inference_times)
            current_avg_total = np.mean(total_sample_times)
            print(f"Processed {i+1}/{len(X_test)} samples - "
                  f"Avg inference: {current_avg_inference*1000:.2f}ms, "
                  f"Avg total: {current_avg_total*1000:.2f}ms")
    total_inference_time = time.time() - total_inference_start 
    timing_stats = {
        'model_load_time': model_load_time,
        'yolo_init_time': yolo_init_time,
        'hog_init_time': hog_init_time,
        'dataset_load_time': dataset_load_time,
        'total_inference_time': total_inference_time,
        'preprocessing_times': preprocessing_times,
        'inference_times': inference_times,
        'postprocessing_times': postprocessing_times,
        'total_sample_times': total_sample_times,
        'detection_times_yolo': detection_times_yolo,
        'detection_times_hog': detection_times_hog,
        'samples_processed': len(X_test)
    }
    avg_preprocessing = np.mean(preprocessing_times) * 1000  # Convert to ms
    avg_inference = np.mean(inference_times) * 1000
    avg_postprocessing = np.mean(postprocessing_times) * 1000
    avg_total_sample = np.mean(total_sample_times) * 1000
    avg_yolo_detection = np.mean(detection_times_yolo) * 1000 if detection_times_yolo else 0
    avg_hog_detection = np.mean(detection_times_hog) * 1000 if detection_times_hog else 0

    print("\n" + "="*80)
    print("TIMING ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total samples processed: {len(X_test)}")
    print(f"Model loading time: {model_load_time:.4f} seconds")
    print(f"Dataset loading time: {dataset_load_time:.4f} seconds")
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print()
    print("Per-sample timing (milliseconds):")
    print("-" * 40)
    print(f"Preprocessing (avg):     {avg_preprocessing:.2f} ± {np.std(preprocessing_times)*1000:.2f} ms")
    print(f"Model inference (avg):   {avg_inference:.2f} ± {np.std(inference_times)*1000:.2f} ms")
    print(f"Postprocessing (avg):    {avg_postprocessing:.2f} ± {np.std(postprocessing_times)*1000:.2f} ms")
    print(f"Total per sample (avg):  {avg_total_sample:.2f} ± {np.std(total_sample_times)*1000:.2f} ms")
    print()
    print("Human detection timing (milliseconds):")
    print("-" * 40)
    print(f"YOLO detection (avg):    {avg_yolo_detection:.2f} ± {np.std(detection_times_yolo)*1000:.2f} ms" if detection_times_yolo else "YOLO detection: N/A")
    print(f"HOG detection (avg):     {avg_hog_detection:.2f} ± {np.std(detection_times_hog)*1000:.2f} ms" if detection_times_hog else "HOG detection: N/A")
    print()
    print("Performance metrics:")
    print("-" * 40)
    print(f"Throughput: {len(X_test)/total_inference_time:.2f} samples/second")
    print(f"FPS equivalent: {1/(avg_total_sample/1000):.2f} FPS")
    print("\n" + "="*80)
    print("MODEL ACCURACY EVALUATION")
    print("="*80)
    
    accuracy = np.mean(np.array(predictions) == y_test) * 100
    print(f"Overall Accuracy: {accuracy:.2f}%")
    class_report = classification_report(y_test, predictions, target_names=label_encoder.classes_)
    print("\nClassification Report:")
    print(class_report)
    
    cm = confusion_matrix(y_test, predictions, labels=label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    confusion_matrix_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_with_timing.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    save_timing_analysis(timing_stats, OUTPUT_DIR)
    save_sampling_analysis(sampling_info_list, OUTPUT_DIR)
    create_timing_visualizations(timing_stats, OUTPUT_DIR)
    print(f"\nGenerating sample predictions with timing info...")
    sample_indices = range(len(X_test))
    
    for idx in sample_indices:
        try:
            video_path = video_paths[idx]
            video_frames, original_frames, sampling_info = sample_frames_action_aware(video_path)
            if original_frames:
                predicted_class = predictions[idx]
                actual_class = y_test[idx]
                confidence = prediction_probabilities[idx][np.argmax(prediction_probabilities[idx])] * 100
                video_filename = f"sample_{idx}_{actual_class}_predicted_{predicted_class}_{confidence:.1f}percent_time_{total_sample_times[idx]*1000:.1f}ms.mp4"
                video_output_path = os.path.join(OUTPUT_DIR, video_filename)
                create_classified_video(original_frames, predicted_class, confidence,
                                      video_output_path, yolo_model, hog_detector)
                motion_viz_path = os.path.join(OUTPUT_DIR, f"motion_analysis_sample_{idx}.png")
                visualize_motion_analysis(sampling_info, motion_viz_path)
                print(f"  Sample {idx}: {actual_class} -> {predicted_class} "
                      f"({confidence:.1f}%) - Processing time: {total_sample_times[idx]*1000:.2f}ms")  
        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
    
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Total processing time: {time.time() - total_inference_start:.2f} seconds")
    print(f"Average inference time per sample: {avg_inference:.2f}ms")
    print(f"Model accuracy: {accuracy:.2f}%")
    return timing_stats, accuracy, class_report

if __name__ == "__main__":
    print("Starting Model Evaluation with Comprehensive Timing Analysis")
    print("=" * 80)
    
    start_time = time.time()
    try:
        result = evaluate_model_with_timing()   
        if result is not None:
            timing_stats, accuracy, class_report = result 
            total_time = time.time() - start_time
            print(f"\nTotal execution time: {total_time:.2f} seconds")
            print(f"Average processing time per sample: {np.mean(timing_stats['total_sample_times'])*1000:.2f}ms")
            print(f"Model accuracy achieved: {accuracy:.2f}%")
        else:
            print("Evaluation failed - no results returned")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    print("\nEvaluation completed!")


def save_timing_analysis(timing_stats, output_dir):
    """Save detailed timing analysis to file"""
    timing_analysis_path = os.path.join(output_dir, "detailed_timing_analysis.txt")  
    with open(timing_analysis_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED TIMING ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
       
        f.write(f"System Information:\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"TensorFlow Version: {tf.__version__}\n")
        f.write(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}\n")
        f.write(f"GPU Devices: {tf.config.list_physical_devices('GPU')}\n\n")
       
        f.write(f"Model Configuration:\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Frame Count: {FRAME_COUNT}\n")
        f.write(f"Frame Size: {FRAME_SIZE}\n")
        f.write(f"Target Duration: {TARGET_DURATION}s\n\n")
       
        f.write(f"Initialization Times:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Loading: {timing_stats['model_load_time']:.4f} seconds\n")
        f.write(f"YOLO Initialization: {timing_stats['yolo_init_time']:.4f} seconds\n")
        f.write(f"HOG Initialization: {timing_stats['hog_init_time']:.4f} seconds\n")
        f.write(f"Dataset Loading: {timing_stats['dataset_load_time']:.4f} seconds\n\n")
       
        f.write(f"Processing Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {timing_stats['samples_processed']}\n")
        f.write(f"Total Inference Time: {timing_stats['total_inference_time']:.4f} seconds\n")
        f.write(f"Throughput: {timing_stats['samples_processed']/timing_stats['total_inference_time']:.2f} samples/second\n\n")
       
        preprocessing_times = np.array(timing_stats['preprocessing_times']) * 1000
        inference_times = np.array(timing_stats['inference_times']) * 1000
        postprocessing_times = np.array(timing_stats['postprocessing_times']) * 1000
        total_sample_times = np.array(timing_stats['total_sample_times']) * 1000

        f.write(f"Per-Sample Timing Statistics (milliseconds):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Stage':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}\n")
        f.write("-" * 70 + "\n")
       
        stages = [
            ("Preprocessing", preprocessing_times),
            ("Model Inference", inference_times),
            ("Postprocessing", postprocessing_times),
            ("Total Sample", total_sample_times)
        ]
        for stage_name, times in stages:
            f.write(f"{stage_name:<20} {np.mean(times):<10.2f} {np.std(times):<10.2f} "
                   f"{np.min(times):<10.2f} {np.max(times):<10.2f} {np.median(times):<10.2f}\n")
        if timing_stats['detection_times_yolo']:
            yolo_times = np.array(timing_stats['detection_times_yolo']) * 1000
            f.write(f"{'YOLO Detection':<20} {np.mean(yolo_times):<10.2f} {np.std(yolo_times):<10.2f} "
                   f"{np.min(yolo_times):<10.2f} {np.max(yolo_times):<10.2f} {np.median(yolo_times):<10.2f}\n")
        if timing_stats['detection_times_hog']:
            hog_times = np.array(timing_stats['detection_times_hog']) * 1000
            f.write(f"{'HOG Detection':<20} {np.mean(hog_times):<10.2f} {np.std(hog_times):<10.2f} "
                   f"{np.min(hog_times):<10.2f} {np.max(hog_times):<10.2f} {np.median(hog_times):<10.2f}\n")
        f.write("\n")
        f.write(f"Inference Time Percentiles (milliseconds):\n")
        f.write("-" * 40 + "\n")
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(inference_times, p)
            f.write(f"P{p}: {value:.2f}ms\n")
       
        f.write(f"\nTotal Sample Time Percentiles (milliseconds):\n")
        f.write("-" * 40 + "\n")
        for p in percentiles:
            value = np.percentile(total_sample_times, p)
            f.write(f"P{p}: {value:.2f}ms\n")
        f.write(f"\nIndividual Sample Times (first 20 samples):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Sample':<8} {'Preprocessing':<15} {'Inference':<12} {'Postprocessing':<15} {'Total':<10}\n")
        f.write("-" * 60 + "\n")
        for i in range(min(20, len(inference_times))):
            f.write(f"{i+1:<8} {preprocessing_times[i]:<15.2f} {inference_times[i]:<12.2f} "
                   f"{postprocessing_times[i]:<15.2f} {total_sample_times[i]:<10.2f}\n")

def create_timing_visualizations(timing_stats, output_dir):
    preprocessing_times = np.array(timing_stats['preprocessing_times']) * 1000
    inference_times = np.array(timing_stats['inference_times']) * 1000
    postprocessing_times = np.array(timing_stats['postprocessing_times']) * 1000
    total_sample_times = np.array(timing_stats['total_sample_times']) * 1000
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Timing Analysis', fontsize=16, fontweight='bold')
    axes[0, 0].hist(inference_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(inference_times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(inference_times):.2f}ms')
    axes[0, 0].set_xlabel('Inference Time (ms)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Inference Times')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    timing_data = [preprocessing_times, inference_times, postprocessing_times, total_sample_times]
    timing_labels = ['Preprocessing', 'Inference', 'Postprocessing', 'Total']
    axes[0, 1].boxplot(timing_data, labels=timing_labels)
    axes[0, 1].set_ylabel('Time (ms)')
    axes[0, 1].set_title('Timing Components Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    sample_indices = range(len(inference_times))
    axes[0, 2].plot(sample_indices, inference_times, alpha=0.7, color='green')
    axes[0, 2].axhline(np.mean(inference_times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(inference_times):.2f}ms')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Inference Time (ms)')
    axes[0, 2].set_title('Inference Time Over Samples')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    components = ['Preprocessing', 'Inference', 'Postprocessing']
    avg_times = [np.mean(preprocessing_times), np.mean(inference_times), np.mean(postprocessing_times)]
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    axes[1, 0].bar(components, avg_times, color=colors)
    axes[1, 0].set_ylabel('Average Time (ms)')
    axes[1, 0].set_title('Average Time per Component')
    axes[1, 0].grid(True, alpha=0.3)
   
    for i, v in enumerate(avg_times):
        axes[1, 0].text(i, v + max(avg_times) * 0.01, f'{v:.2f}ms',
                       ha='center', va='bottom', fontweight='bold')
   
    sorted_times = np.sort(inference_times)
    percentiles = np.arange(1, 101)
    axes[1, 1].plot(percentiles, np.percentile(sorted_times, percentiles), color='purple')
    axes[1, 1].set_xlabel('Percentile')
    axes[1, 1].set_ylabel('Inference Time (ms)')
    axes[1, 1].set_title('Cumulative Distribution of Inference Times')
    axes[1, 1].grid(True, alpha=0.3)
   
    key_percentiles = [50, 90, 95, 99]
    for p in key_percentiles:
        value = np.percentile(inference_times, p)
        axes[1, 1].axhline(value, color='red', linestyle=':', alpha=0.7)
        axes[1, 1].text(p, value, f'P{p}: {value:.1f}ms',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    axes[1, 2].axis('off')
    summary_text = f"""
Performance Summary

Average Times:
- Preprocessing: {np.mean(preprocessing_times):.2f} ± {np.std(preprocessing_times):.2f} ms
- Model Inference: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms  
- Postprocessing: {np.mean(postprocessing_times):.2f} ± {np.std(postprocessing_times):.2f} ms
- Total per Sample: {np.mean(total_sample_times):.2f} ± {np.std(total_sample_times):.2f} ms

Throughput: {timing_stats['samples_processed']/timing_stats['total_inference_time']:.2f} samples/sec
FPS Equivalent: {1/(np.mean(total_sample_times)/1000):.1f} FPS
Total Samples: {timing_stats['samples_processed']}
Total Time: {timing_stats['total_inference_time']:.2f} seconds
    """
    axes[1, 2].text(0, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    plt.tight_layout()
    timing_viz_path = os.path.join(output_dir, 'comprehensive_timing_analysis.png')
    plt.savefig(timing_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
   
    if timing_stats['detection_times_yolo'] or timing_stats['detection_times_hog']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Human Detection Timing Analysis', fontsize=14, fontweight='bold')
        if timing_stats['detection_times_yolo']:
            yolo_times = np.array(timing_stats['detection_times_yolo']) * 1000
            axes[0].hist(yolo_times, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[0].axvline(np.mean(yolo_times), color='red', linestyle='--',
                           label=f'Mean: {np.mean(yolo_times):.2f}ms')
            axes[0].set_xlabel('YOLO Detection Time (ms)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('YOLO Detection Time Distribution')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        if timing_stats['detection_times_hog']:
            hog_times = np.array(timing_stats['detection_times_hog']) * 1000
            axes[1].hist(hog_times, bins=20, alpha=0.7, color='cyan', edgecolor='black')
            axes[1].axvline(np.mean(hog_times), color='red', linestyle='--',
                           label=f'Mean: {np.mean(hog_times):.2f}ms')
            axes[1].set_xlabel('HOG Detection Time (ms)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('HOG Detection Time Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        detection_viz_path = os.path.join(output_dir, 'detection_timing_analysis.png')
        plt.savefig(detection_viz_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("Starting Model Evaluation with Comprehensive Timing Analysis")
    print("=" * 80)
    start_time = time.time()
    try:
        timing_stats, accuracy, class_report = evaluate_model_with_timing()
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        print(f"Average processing time per sample: {np.mean(timing_stats['total_sample_times'])*1000:.2f}ms")
        print(f"Model accuracy achieved: {accuracy:.2f}%")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    print("\nEvaluation completed!")



