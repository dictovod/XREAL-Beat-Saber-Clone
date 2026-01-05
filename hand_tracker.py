#!/usr/bin/env python3
"""
XREAL Hand Tracker
Отслеживание рук через камеру XREAL One Pro с использованием MediaPipe Vision
"""

import cv2
import numpy as np
import socket
import threading
import time
from collections import deque
from typing import Optional, Tuple, List
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Параметры камеры XREAL
GLASSES_IP = "169.254.2.1"
VIDEO_PORT = 52997
PACKET_SIZE = 193862
HEADER_OFFSET = 0x140
WIDTH = 512
HEIGHT = 378
IMAGE_SIZE = WIDTH * HEIGHT


class HandLandmark:
    """Ключевые точки руки (индексы как в MediaPipe)"""
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20


class Hand:
    """Данные одной руки"""
    def __init__(self, label: str, landmarks: np.ndarray, handedness_score: float):
        self.label = label  # 'Left' или 'Right'
        self.landmarks = landmarks  # Массив из 21 точки (x, y, z) - нормализованные координаты
        self.handedness_score = handedness_score
        self.timestamp = time.time()
        
        # Вычисляем позицию запястья
        self.wrist_pos = landmarks[HandLandmark.WRIST]
        
        # Вычисляем центр ладони
        palm_landmarks = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        self.palm_center = np.mean(palm_landmarks, axis=0)
        
        # Скорость (будет вычислена позже)
        self.velocity = np.array([0.0, 0.0, 0.0])
    
    def get_finger_tip(self, finger_id: int) -> np.ndarray:
        """Получить позицию кончика пальца"""
        return self.landmarks[finger_id]
    
    def is_fist(self) -> bool:
        """Проверка, сжата ли рука в кулак"""
        distances = []
        for tip_id in [HandLandmark.INDEX_TIP, HandLandmark.MIDDLE_TIP, 
                       HandLandmark.RING_TIP, HandLandmark.PINKY_TIP]:
            tip = self.landmarks[tip_id]
            dist = np.linalg.norm(tip - self.palm_center)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        return avg_distance < 0.15
    
    def is_pointing(self) -> bool:
        """Проверка, указывает ли рука пальцем"""
        index_tip = self.landmarks[HandLandmark.INDEX_TIP]
        index_base = self.landmarks[5]
        
        index_extended = np.linalg.norm(index_tip - index_base) > 0.2
        
        other_tips = [HandLandmark.MIDDLE_TIP, HandLandmark.RING_TIP, HandLandmark.PINKY_TIP]
        other_distances = [np.linalg.norm(self.landmarks[tip] - self.palm_center) 
                          for tip in other_tips]
        others_bent = all(d < 0.15 for d in other_distances)
        
        return index_extended and others_bent


class XrealHandTracker:
    """
    Трекер рук для XREAL камеры
    Использует MediaPipe Vision API для детекции рук
    """
    
    def __init__(self):
        # MediaPipe Hand Landmarker настройки
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Подключение к камере
        self.sock = None
        self.connected = False
        self.running = False
        self.receive_thread = None
        
        # Буферы
        self.frame_buffer = deque(maxlen=2)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Отслеживание рук
        self.left_hand: Optional[Hand] = None
        self.right_hand: Optional[Hand] = None
        self.hands_lock = threading.Lock()
        
        # История для вычисления скорости
        self.left_hand_history = deque(maxlen=5)
        self.right_hand_history = deque(maxlen=5)
        
        # Статистика
        self.frames_received = 0
        self.frames_processed = 0
        self.detection_count = 0
        self.start_time = time.time()
        self.frame_timestamp_ms = 0
        
    def connect(self) -> bool:
        """Подключиться к камере XREAL"""
        try:
            logger.info(f"Connecting to XREAL camera at {GLASSES_IP}:{VIDEO_PORT}...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((GLASSES_IP, VIDEO_PORT))
            self.sock.settimeout(1)
            self.connected = True
            logger.info("✓ Camera connected!")
            return True
        except Exception as e:
            logger.error(f"✗ Camera connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Отключиться от камеры"""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False
        logger.info("Camera disconnected")
    
    def decode_frame(self, packet: bytes) -> Optional[np.ndarray]:
        """Декодировать видеопакет в изображение"""
        if len(packet) < HEADER_OFFSET + IMAGE_SIZE:
            return None
        
        data = packet[HEADER_OFFSET:HEADER_OFFSET + IMAGE_SIZE]
        pixels = np.frombuffer(data, dtype=np.uint8)
        pixels = ((pixels >> 4) & 0x0F) * 17
        img = pixels.reshape((HEIGHT, WIDTH))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img_bgr
    
    def receive_loop(self):
        """Поток приёма видеоданных"""
        buffer = b''
        
        while self.running and self.connected:
            try:
                data = self.sock.recv(65536)
                if not data:
                    continue
                
                buffer += data
                
                while len(buffer) >= PACKET_SIZE:
                    packet = buffer[:PACKET_SIZE]
                    buffer = buffer[PACKET_SIZE:]
                    
                    frame = self.decode_frame(packet)
                    if frame is not None:
                        self.frames_received += 1
                        with self.frame_lock:
                            self.frame_buffer.append(frame)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Receive error: {e}")
                break
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Hand], Optional[Hand]]:
        """
        Обработать кадр и обнаружить руки
        Возвращает (левая_рука, правая_рука)
        """
        # Конвертируем BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Создаём MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Детекция рук
        self.frame_timestamp_ms += 33  # ~30 FPS
        detection_result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        left_hand = None
        right_hand = None
        
        if detection_result.hand_landmarks and detection_result.handedness:
            for hand_landmarks, handedness in zip(detection_result.hand_landmarks, 
                                                   detection_result.handedness):
                # Получаем метку руки
                label = handedness[0].category_name  # 'Left' или 'Right'
                score = handedness[0].score
                
                # Конвертируем landmarks в numpy массив
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                
                # Создаём объект Hand
                hand = Hand(label, landmarks, score)
                
                # Сохраняем
                if label == 'Left':
                    left_hand = hand
                else:
                    right_hand = hand
        
        return left_hand, right_hand
    
    def calculate_velocity(self, current: Optional[Hand], 
                          history: deque) -> Optional[np.ndarray]:
        """Вычислить скорость движения руки"""
        if current is None or len(history) < 2:
            return None
        
        positions = [h.wrist_pos for h in history]
        times = [h.timestamp for h in history]
        
        dt = times[-1] - times[0]
        if dt < 0.01:
            return np.array([0.0, 0.0, 0.0])
        
        displacement = positions[-1] - positions[0]
        velocity = displacement / dt
        
        return velocity
    
    def update(self):
        """Обновить состояние трекера"""
        with self.frame_lock:
            if not self.frame_buffer:
                return
            frame = self.frame_buffer[-1]
            self.current_frame = frame.copy()
        
        left, right = self.process_frame(frame)
        self.frames_processed += 1
        
        if left:
            self.left_hand_history.append(left)
            left.velocity = self.calculate_velocity(left, self.left_hand_history) or np.array([0, 0, 0])
        
        if right:
            self.right_hand_history.append(right)
            right.velocity = self.calculate_velocity(right, self.right_hand_history) or np.array([0, 0, 0])
        
        with self.hands_lock:
            self.left_hand = left
            self.right_hand = right
            if left or right:
                self.detection_count += 1
    
    def get_hands(self) -> Tuple[Optional[Hand], Optional[Hand]]:
        """Получить текущее состояние рук (thread-safe)"""
        with self.hands_lock:
            return self.left_hand, self.right_hand
    
    def get_debug_frame(self) -> Optional[np.ndarray]:
        """Получить кадр с визуализацией для отладки"""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        left, right = self.get_hands()
        
        if left:
            self._draw_hand_on_frame(frame, left, (0, 0, 255))
        
        if right:
            self._draw_hand_on_frame(frame, right, (255, 0, 0))
        
        elapsed = max(1, time.time() - self.start_time)
        cv2.putText(frame, f"FPS: {self.frames_received / elapsed:.0f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {self.detection_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _draw_hand_on_frame(self, frame: np.ndarray, hand: Hand, color: Tuple[int, int, int]):
        """Нарисовать руку на кадре"""
        h, w = frame.shape[:2]
        
        # Соединения MediaPipe
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start_idx, end_idx in connections:
            start = hand.landmarks[start_idx]
            end = hand.landmarks[end_idx]
            
            start_point = (int(start[0] * w), int(start[1] * h))
            end_point = (int(end[0] * w), int(end[1] * h))
            
            cv2.line(frame, start_point, end_point, color, 2)
        
        for landmark in hand.landmarks:
            point = (int(landmark[0] * w), int(landmark[1] * h))
            cv2.circle(frame, point, 3, color, -1)
        
        wrist = hand.wrist_pos
        wrist_point = (int(wrist[0] * w), int(wrist[1] * h))
        cv2.circle(frame, wrist_point, 8, color, 3)
        
        cv2.putText(frame, hand.label, wrist_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        speed = np.linalg.norm(hand.velocity[:2])
        if speed > 0.1:
            cv2.putText(frame, f"V: {speed:.2f}", 
                       (wrist_point[0], wrist_point[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def start(self) -> bool:
        """Запустить трекер"""
        if not self.connect():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        self.receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.receive_thread.start()
        
        logger.info("Hand tracker started")
        return True
    
    def stop(self):
        """Остановить трекер"""
        self.running = False
        self.disconnect()
        
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        
        self.detector.close()
        logger.info("Hand tracker stopped")


# =============================================================================
# Тестирование
# =============================================================================

def main():
    """Тест трекера с визуализацией"""
    print("=" * 60)
    print("XREAL Hand Tracker Test (MediaPipe Vision)")
    print("=" * 60)
    print("Press 'q' to quit")
    print()
    
    tracker = XrealHandTracker()
    
    if not tracker.start():
        print("Failed to start tracker")
        return
    
    try:
        while True:
            tracker.update()
            
            debug_frame = tracker.get_debug_frame()
            if debug_frame is not None:
                cv2.imshow("XREAL Hand Tracking", debug_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
        
        print(f"\nStatistics:")
        print(f"  Frames received: {tracker.frames_received}")
        print(f"  Frames processed: {tracker.frames_processed}")
        print(f"  Hands detected: {tracker.detection_count}")


if __name__ == "__main__":
    main()
