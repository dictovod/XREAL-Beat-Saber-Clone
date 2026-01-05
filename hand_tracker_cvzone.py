#!/usr/bin/env python3
"""
XREAL Hand Tracker
Отслеживание рук через камеру XREAL One Pro с использованием детекции движения
"""

import cv2
import numpy as np
import socket
import threading
import time
from collections import deque
from typing import Optional, Tuple, List
import logging

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


class MotionHandDetector:
    """Детектор рук на основе движения"""
    
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        self.prev_frame = None
        self.motion_threshold = 500  # Минимальная площадь движения
        
    def detect_hands(self, frame: np.ndarray) -> List[dict]:
        """
        Детекция рук по движению
        Возвращает список словарей с ключами: 'type', 'center', 'area'
        """
        h, w = frame.shape[:2]
        
        # Применяем background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Дополнительная фильтрация: frame differencing
        if self.prev_frame is not None:
            # Конвертируем в grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Frame difference
            frame_diff = cv2.absdiff(gray, prev_gray)
            _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Комбинируем с fg_mask
            combined_mask = cv2.bitwise_or(fg_mask, diff_thresh)
        else:
            combined_mask = fg_mask
        
        self.prev_frame = frame.copy()
        
        # Морфологические операции для очистки
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        # Найти контуры движения
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        
        # Берём контуры достаточного размера
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:
                valid_contours.append((contour, area))
        
        # Сортируем по площади и берём 2 самых больших
        valid_contours = sorted(valid_contours, key=lambda x: x[1], reverse=True)[:2]
        
        for contour, area in valid_contours:
            # Получаем bbox
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Центр движущейся области
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
                'area': area,
                'contour': contour
            })
        
        return hands
    
    def create_landmarks(self, hand_data: dict) -> np.ndarray:
        """
        Создать landmarks на основе центра руки
        Для Beat Saber достаточно запястья в центре
        """
        cx, cy = hand_data['center']
        
        # Создаём 21 точку, все в центре (запястье)
        landmarks = np.zeros((21, 3), dtype=np.float32)
        
        # Все точки сходятся к центру руки
        for i in range(21):
            landmarks[i] = [cx, cy, 0]
        
        return landmarks


class XrealHandTracker:
    """
    Трекер рук для XREAL камеры
    Использует детекцию движения
    """
    
    def __init__(self):
        # Детектор на основе движения
        self.detector = MotionHandDetector()
        
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
        # Детекция рук
        hands_data = self.detector.detect_hands(frame)
        
        left_hand = None
        right_hand = None
        
        for hand_data in hands_data:
            hand_type = hand_data['type']
            
            # Создаём landmarks
            landmarks = self.detector.create_landmarks(hand_data)
            
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
        
        # Рисуем только контуры запястий
        h, w = frame.shape[:2]
        
        if left:
            wrist = left.wrist_pos
            wrist_point = (int(wrist[0] * w), int(wrist[1] * h))
            
            # Красный круг для левой руки
            cv2.circle(frame, wrist_point, 40, (0, 0, 255), 3)
            cv2.circle(frame, wrist_point, 10, (0, 0, 255), -1)
            cv2.putText(frame, "LEFT", (wrist_point[0] - 40, wrist_point[1] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Скорость
            speed = np.linalg.norm(left.velocity[:2])
            if speed > 0.5:
                cv2.putText(frame, f"V: {speed:.1f}", 
                           (wrist_point[0] - 40, wrist_point[1] + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if right:
            wrist = right.wrist_pos
            wrist_point = (int(wrist[0] * w), int(wrist[1] * h))
            
            # Синий круг для правой руки
            cv2.circle(frame, wrist_point, 40, (255, 0, 0), 3)
            cv2.circle(frame, wrist_point, 10, (255, 0, 0), -1)
            cv2.putText(frame, "RIGHT", (wrist_point[0] - 40, wrist_point[1] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Скорость
            speed = np.linalg.norm(right.velocity[:2])
            if speed > 0.5:
                cv2.putText(frame, f"V: {speed:.1f}", 
                           (wrist_point[0] - 40, wrist_point[1] + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Статистика
        elapsed = max(1, time.time() - self.start_time)
        cv2.putText(frame, f"FPS: {self.frames_received / elapsed:.0f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Hands: {self.detection_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "MOVE YOUR HANDS!", 
                   (w//2 - 120, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return frame
    
    def start(self) -> bool:
        """Запустить трекер"""
        if not self.connect():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        self.receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
        self.receive_thread.start()
        
        logger.info("Hand tracker started (Motion Detection)")
        logger.info(">>> WAVE YOUR HANDS to see them detected! <<<")
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
    print("XREAL Hand Tracker Test (Motion Detection)")
    print("=" * 60)
    print(">>> WAVE YOUR HANDS to see them detected! <<<")
    print("Press 'q' to quit")
    print()
    
    tracker = XrealHandTracker()
    
    if not tracker.start():
        print("Failed to start tracker")
        return
    
    print("\nWait 2 seconds for background calibration...")
    time.sleep(2)
    print("NOW WAVE YOUR HANDS!")
    
    try:
        while True:
            tracker.update()
            
            debug_frame = tracker.get_debug_frame()
            if debug_frame is not None:
                cv2.imshow("XREAL Hand Tracking - WAVE YOUR HANDS!", debug_frame)
            
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
