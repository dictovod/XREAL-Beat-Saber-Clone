#!/usr/bin/env python3
"""
XREAL Hand Tracker
Отслеживание рук через камеру XREAL One Pro с использованием YOLOv8-pose
"""

import cv2
import numpy as np
import socket
import threading
import time
from collections import deque
from typing import Optional, Tuple, List
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")

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
        self.landmarks = landmarks  # Массив из 21 точки (x, y, z)
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


class SimpleHandDetector:
    """Упрощённый детектор рук на основе цвета кожи и контуров"""
    
    def __init__(self):
        # Диапазоны цвета кожи в HSV
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    def detect_hands(self, frame: np.ndarray) -> List[dict]:
        """
        Простая детекция рук по цвету кожи
        Возвращает список словарей с ключами: 'type', 'center', 'bbox'
        """
        h, w = frame.shape[:2]
        
        # Конвертируем в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Маска кожи
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Найти контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        
        # Берём 2 самых больших контура
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Фильтруем маленькие области
            if area < 1000:
                continue
            
            # Получаем bbox
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Центр руки
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + bw // 2, y + bh // 2
            
            # Определяем левую/правую руку по позиции
            hand_type = 'Left' if cx < w // 2 else 'Right'
            
            hands.append({
                'type': hand_type,
                'center': (cx / w, cy / h),  # Нормализованные координаты
                'bbox': (x, y, bw, bh),
                'contour': contour
            })
        
        return hands
    
    def create_fake_landmarks(self, hand_data: dict, img_width: int, img_height: int) -> np.ndarray:
        """
        Создать примерные landmarks на основе центра руки
        Для Beat Saber нужны только приблизительные позиции
        """
        cx, cy = hand_data['center']
        
        # Создаём 21 точку в виде сетки вокруг центра
        landmarks = np.zeros((21, 3), dtype=np.float32)
        
        # Запястье в центре
        landmarks[0] = [cx, cy, 0]
        
        # Пальцы расходятся от центра
        offsets = [
            # Большой палец
            [(-0.08, -0.06), (-0.10, -0.04), (-0.12, -0.02), (-0.14, 0)],
            # Указательный
            [(-0.04, -0.08), (-0.04, -0.12), (-0.04, -0.16), (-0.04, -0.20)],
            # Средний
            [(0, -0.08), (0, -0.12), (0, -0.16), (0, -0.20)],
            # Безымянный
            [(0.04, -0.08), (0.04, -0.12), (0.04, -0.16), (0.04, -0.20)],
            # Мизинец
            [(0.08, -0.06), (0.08, -0.10), (0.08, -0.14), (0.08, -0.18)]
        ]
        
        idx = 1
        for finger in offsets:
            for dx, dy in finger:
                landmarks[idx] = [cx + dx, cy + dy, 0]
                idx += 1
        
        return landmarks


class XrealHandTracker:
    """
    Трекер рук для XREAL камеры
    Использует простую детекцию на основе цвета кожи
    """
    
    def __init__(self):
        # Простой детектор
        self.detector = SimpleHandDetector()
        
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
        h, w = frame.shape[:2]
        
        # Детекция рук
        hands_data = self.detector.detect_hands(frame)
        
        left_hand = None
        right_hand = None
        
        for hand_data in hands_data:
            hand_type = hand_data['type']
            
            # Создаём примерные landmarks
            landmarks = self.detector.create_fake_landmarks(hand_data, w, h)
            
            # Создаём объект Hand
            hand = Hand(hand_type, landmarks, 1.0)
            
            if hand_type == 'Left':
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
        
        # Рисуем только центр руки (запястье) и метку
        wrist = hand.wrist_pos
        wrist_point = (int(wrist[0] * w), int(wrist[1] * h))
        
        # Большой круг для руки
        cv2.circle(frame, wrist_point, 30, color, 3)
        cv2.circle(frame, wrist_point, 8, color, -1)
        
        # Метка руки
        cv2.putText(frame, hand.label, 
                   (wrist_point[0] - 30, wrist_point[1] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Скорость
        speed = np.linalg.norm(hand.velocity[:2])
        if speed > 0.1:
            cv2.putText(frame, f"V: {speed:.2f}", 
                       (wrist_point[0] - 30, wrist_point[1] + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def start(self) -> bool:
        """Запустить трекер"""
        if not self.connect():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        self.receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.receive_thread.start()
        
        logger.info("Hand tracker started (Simple Color-based Detection)")
        return True
    
    def stop(self):
        """Остановить трекер"""
        self.running = False
        self.disconnect()
        
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        
        logger.info("Hand tracker stopped")


# =============================================================================
# Тестирование
# =============================================================================

def main():
    """Тест трекера с визуализацией"""
    print("=" * 60)
    print("XREAL Hand Tracker Test (Simple Detection)")
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
