#!/usr/bin/env python3
"""
XREAL Beat Saber Clone
Ритм-игра с управлением движениями рук через гироскоп очков

Управление:
- Движения рук (через гироскоп) для отбивания кубиков
- ESC для выхода
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import math
import random
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imu_reader import ImuReader, ImuData, ConnectionState
from config import GLASSES_IP_PRIMARY, PORT_IMU


class Cube:
    """Летящий кубик"""
    def __init__(self, lane, color, spawn_time):
        self.lane = lane  # -1, 0, 1 (левый, центр, правый)
        self.color = color  # (r, g, b)
        self.z = -50  # Начальная позиция вдали
        self.spawn_time = spawn_time
        self.hit = False
        self.size = 1.0
        self.hit_time = 0
        
    def update(self, dt):
        """Обновить позицию"""
        self.z += 20 * dt  # Скорость приближения
        
        # Анимация попадания
        if self.hit:
            age = time.time() - self.hit_time
            self.size = 1.0 + age * 2  # Увеличиваемся
        
    def is_in_hit_zone(self):
        """Проверка в зоне удара"""
        return -3 < self.z < 1
    
    def is_missed(self):
        """Проверка промаха"""
        return self.z > 5
    
    def draw(self):
        """Отрисовать кубик"""
        glPushMatrix()
        
        # Позиция
        x_pos = self.lane * 3.5
        glTranslatef(x_pos, 0, self.z)
        
        # Вращение для красоты
        angle = self.spawn_time * 50
        glRotatef(angle, 0, 1, 0)
        
        # Цвет кубика
        if self.hit:
            alpha = max(0, 1 - (time.time() - self.hit_time) * 2)
            glColor4f(*self.color, alpha * 0.5)
        else:
            glColor4f(*self.color, 0.9)
        
        # Рисуем куб
        size = self.size
        vertices = [
            [-size, -size, -size], [size, -size, -size],
            [size, size, -size], [-size, size, -size],
            [-size, -size, size], [size, -size, size],
            [size, size, size], [-size, size, size]
        ]
        
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
        faces = [
            (0,1,2,3), (4,5,6,7),
            (0,1,5,4), (2,3,7,6),
            (0,3,7,4), (1,2,6,5)
        ]
        
        # Заполнение граней
        glEnable(GL_BLEND)
        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()
        
        # Рёбра (контур)
        glDisable(GL_LIGHTING)
        if not self.hit:
            glColor3f(0.1, 0.1, 0.1)
        else:
            glColor4f(1, 1, 1, 0.3)
        glLineWidth(2)
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
        glEnable(GL_LIGHTING)
        
        glPopMatrix()


class HandSlash:
    """Визуальный эффект удара рукой"""
    def __init__(self, side, position):
        self.side = side  # 'left' или 'right'
        self.position = position  # (x, y, z)
        self.created_time = time.time()
        self.lifetime = 0.3  # Секунды
        
    def is_alive(self):
        return time.time() - self.created_time < self.lifetime
    
    def draw(self):
        age = time.time() - self.created_time
        alpha = 1 - (age / self.lifetime)
        
        if alpha <= 0:
            return
        
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(*self.position)
        
        # Цвет в зависимости от руки
        if self.side == 'left':
            glColor4f(0, 0.5, 1, alpha)
        else:
            glColor4f(1, 0.3, 0, alpha)
        
        # Рисуем "след" удара - расширяющиеся линии
        glLineWidth(5)
        glBegin(GL_LINES)
        spread = age * 10
        glVertex3f(-spread, 0, 0)
        glVertex3f(spread, 0, 0)
        glVertex3f(0, -spread, 0)
        glVertex3f(0, spread, 0)
        glEnd()
        
        glPopMatrix()
        glEnable(GL_LIGHTING)


class BeatSaberGame:
    """Основной класс игры"""
    
    def __init__(self):
        # Pygame и OpenGL
        pygame.init()
        
        # ПОЛНОЭКРАННЫЙ РЕЖИМ
        display_info = pygame.display.Info()
        self.width = display_info.current_w
        self.height = display_info.current_h
        
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            DOUBLEBUF | OPENGL | FULLSCREEN
        )
        pygame.display.set_caption("XREAL Beat Saber")
        
        # Скрыть курсор мыши
        pygame.mouse.set_visible(False)
        
        # OpenGL настройки
        self.setup_opengl()
        
        # IMU данные для отслеживания рук
        self.imu_reader = None
        self.gyro_history = []
        self.max_history = 5
        
        # Отслеживание ударов руками
        self.left_hand_velocity = 0
        self.right_hand_velocity = 0
        self.last_left_hit = 0
        self.last_right_hit = 0
        self.hit_cooldown = 0.2  # Секунды между ударами
        
        # Игровая логика
        self.cubes = []
        self.slashes = []  # Визуальные эффекты ударов
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.missed = 0
        self.hits = 0
        self.game_time = 0
        self.spawn_timer = 0
        self.spawn_interval = 1.2  # Секунды между спавном
        
        # Шрифт
        try:
            self.font = pygame.font.Font(None, 72)
            self.small_font = pygame.font.Font(None, 48)
            self.tiny_font = pygame.font.Font(None, 36)
        except:
            self.font = pygame.font.SysFont('Arial', 72)
            self.small_font = pygame.font.SysFont('Arial', 48)
            self.tiny_font = pygame.font.SysFont('Arial', 36)
        
        # Состояние
        self.running = False
        self.connected = False
        self.paused = False
        self.show_instructions = True
        self.instruction_timer = 5.0  # Показывать инструкции 5 секунд
        
    def setup_opengl(self):
        """Настройка OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Перспектива
        glMatrixMode(GL_PROJECTION)
        gluPerspective(60, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Освещение
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 10, 10, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.5, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])
        
        # Фоновый цвет
        glClearColor(0.05, 0.05, 0.15, 1.0)
        
    def connect_imu(self):
        """Подключение к IMU"""
        print(f"Connecting to XREAL IMU at {GLASSES_IP_PRIMARY}:{PORT_IMU}...")
        
        self.imu_reader = ImuReader(
            host=GLASSES_IP_PRIMARY,
            port=PORT_IMU,
            on_state_change=self.on_imu_state_change,
            auto_reconnect=True
        )
        
        self.imu_reader.start()
        
    def on_imu_state_change(self, state: ConnectionState):
        """Обработка изменения состояния IMU"""
        if state == ConnectionState.CONNECTED:
            self.connected = True
            print("✓ IMU connected! Ready to play!")
        else:
            self.connected = False
            if state == ConnectionState.ERROR:
                print("✗ IMU connection error!")
    
    def detect_hand_strikes(self):
        """Определить удары руками по данным гироскопа"""
        if not self.imu_reader:
            return None, None
        
        imu = self.imu_reader.get_latest()
        if not imu:
            return None, None
        
        # Добавляем в историю
        self.gyro_history.append(imu)
        if len(self.gyro_history) > self.max_history:
            self.gyro_history.pop(0)
        
        if len(self.gyro_history) < 3:
            return None, None
        
        current_time = time.time()
        left_strike = None
        right_strike = None
        
        # Анализируем движения
        # Левая рука: быстрое движение вправо (положительный gyro_z)
        # Правая рука: быстрое движение влево (отрицательный gyro_z)
        
        recent = self.gyro_history[-3:]
        
        # Левая рука (свайп вправо)
        left_velocities = [g.gyro_z for g in recent]
        left_speed = sum(left_velocities) / len(left_velocities)
        
        if left_speed > 4.0 and current_time - self.last_left_hit > self.hit_cooldown:
            self.last_left_hit = current_time
            left_strike = 'left'
            print(f"LEFT HAND STRIKE! Speed: {left_speed:.1f}")
        
        # Правая рука (свайп влево)
        right_speed = -left_speed  # Инвертируем для правой руки
        
        if right_speed > 4.0 and current_time - self.last_right_hit > self.hit_cooldown:
            self.last_right_hit = current_time
            right_strike = 'right'
            print(f"RIGHT HAND STRIKE! Speed: {right_speed:.1f}")
        
        # Альтернативно: быстрое движение вверх/вниз (gyro_x) для обеих рук
        vertical_velocities = [abs(g.gyro_x) for g in recent]
        vertical_speed = sum(vertical_velocities) / len(vertical_velocities)
        
        if vertical_speed > 5.0:
            if current_time - self.last_left_hit > self.hit_cooldown:
                self.last_left_hit = current_time
                left_strike = 'left'
            if current_time - self.last_right_hit > self.hit_cooldown:
                self.last_right_hit = current_time
                right_strike = 'right'
        
        return left_strike, right_strike
    
    def spawn_cube(self):
        """Создать новый кубик"""
        lane = random.choice([-1, 0, 1])
        
        # Цвета: красный (левая рука), синий (правая рука), зелёный (любая)
        color_choices = [
            ((1, 0.2, 0.2), 'left'),    # Красный - левая
            ((0.2, 0.4, 1), 'right'),   # Синий - правая
            ((0.2, 1, 0.2), 'both'),    # Зелёный - любая
        ]
        color, hand = random.choice(color_choices)
        
        cube = Cube(lane, color, self.game_time)
        cube.required_hand = hand
        self.cubes.append(cube)
    
    def check_hits(self, left_strike, right_strike):
        """Проверить попадания по кубикам"""
        if not left_strike and not right_strike:
            return
        
        for cube in self.cubes:
            if cube.hit:
                continue
            
            # Кубик в зоне удара
            if not cube.is_in_hit_zone():
                continue
            
            # Проверяем соответствие руки
            hit = False
            hand_used = None
            
            if cube.required_hand == 'left' and left_strike:
                hit = True
                hand_used = 'left'
            elif cube.required_hand == 'right' and right_strike:
                hit = True
                hand_used = 'right'
            elif cube.required_hand == 'both' and (left_strike or right_strike):
                hit = True
                hand_used = left_strike if left_strike else right_strike
            
            if hit:
                cube.hit = True
                cube.hit_time = time.time()
                self.score += 100 * (self.combo + 1)
                self.combo += 1
                self.max_combo = max(self.max_combo, self.combo)
                self.hits += 1
                
                # Визуальный эффект
                x_pos = cube.lane * 3.5
                slash = HandSlash(hand_used, (x_pos, 0, cube.z))
                self.slashes.append(slash)
                
                print(f"HIT! Score: {self.score}, Combo: x{self.combo}")
                break  # Один удар за раз
    
    def update(self, dt):
        """Обновить игровое состояние"""
        if self.paused:
            return
        
        self.game_time += dt
        
        # Убрать инструкции через 5 секунд
        if self.show_instructions:
            self.instruction_timer -= dt
            if self.instruction_timer <= 0:
                self.show_instructions = False
        
        # Определить удары руками
        left_strike, right_strike = self.detect_hand_strikes()
        
        # Спавн кубиков
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self.spawn_cube()
            # Постепенное ускорение
            self.spawn_interval = max(0.6, 1.2 - self.game_time * 0.015)
        
        # Обновить кубики
        for cube in self.cubes[:]:
            cube.update(dt)
            
            # Удалить пропущенные
            if cube.is_missed():
                self.cubes.remove(cube)
                if not cube.hit:
                    self.missed += 1
                    self.combo = 0
                    print(f"MISS! Total missed: {self.missed}")
            
            # Удалить попавшие (после анимации)
            if cube.hit and (time.time() - cube.hit_time) > 0.5:
                self.cubes.remove(cube)
        
        # Проверить попадания
        self.check_hits(left_strike, right_strike)
        
        # Обновить визуальные эффекты
        self.slashes = [s for s in self.slashes if s.is_alive()]
    
    def draw_3d(self):
        """Отрисовка 3D сцены"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Камера
        gluLookAt(
            0, 3, 8,    # Позиция камеры (выше и дальше)
            0, 0, -10,  # Точка взгляда (вперёд)
            0, 1, 0     # Вектор "вверх"
        )
        
        # Пол (сетка)
        self.draw_floor()
        
        # Дорожки
        self.draw_lanes()
        
        # Кубики
        for cube in self.cubes:
            cube.draw()
        
        # Визуальные эффекты ударов
        for slash in self.slashes:
            slash.draw()
        
        # Индикатор зоны удара
        self.draw_hit_zone()
    
    def draw_floor(self):
        """Рисовать пол с сеткой"""
        glDisable(GL_LIGHTING)
        glColor3f(0.1, 0.1, 0.2)
        glLineWidth(1)
        
        glBegin(GL_LINES)
        for i in range(-25, 10, 2):
            # Линии по Z
            glVertex3f(-15, -3, i)
            glVertex3f(15, -3, i)
        
        for i in range(-7, 8, 2):
            # Линии по X
            glVertex3f(i*2, -3, -25)
            glVertex3f(i*2, -3, 10)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_lanes(self):
        """Рисовать дорожки с подсветкой"""
        glDisable(GL_LIGHTING)
        
        for lane in [-1, 0, 1]:
            # Цвет дорожки
            if lane == -1:
                glColor4f(1, 0.2, 0.2, 0.15)  # Красная - левая
            elif lane == 1:
                glColor4f(0.2, 0.4, 1, 0.15)  # Синяя - правая
            else:
                glColor4f(0.2, 1, 0.2, 0.15)  # Зелёная - центр
            
            glBegin(GL_QUADS)
            x = lane * 3.5
            glVertex3f(x - 1.5, -2.99, -50)
            glVertex3f(x + 1.5, -2.99, -50)
            glVertex3f(x + 1.5, -2.99, 10)
            glVertex3f(x - 1.5, -2.99, 10)
            glEnd()
            
            # Границы дорожек
            glColor4f(0.5, 0.5, 0.7, 0.5)
            glLineWidth(2)
            glBegin(GL_LINES)
            glVertex3f(x - 1.5, -2.98, -50)
            glVertex3f(x - 1.5, -2.98, 10)
            glVertex3f(x + 1.5, -2.98, -50)
            glVertex3f(x + 1.5, -2.98, 10)
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_hit_zone(self):
        """Рисовать зону удара"""
        glDisable(GL_LIGHTING)
        glColor4f(1, 1, 0, 0.3)
        glLineWidth(3)
        
        # Линия зоны удара
        glBegin(GL_LINES)
        glVertex3f(-12, -2.9, 0)
        glVertex3f(12, -2.9, 0)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_hud(self):
        """Рисовать HUD (2D overlay)"""
        # Переключаемся в 2D режим
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Счёт (верхний левый угол)
        self.render_text(f"SCORE: {self.score}", 30, 30, self.font, (255, 255, 255))
        
        # Комбо (если есть)
        if self.combo > 0:
            combo_size = min(100, 72 + self.combo * 2)  # Увеличивается с комбо
            combo_font = pygame.font.Font(None, combo_size)
            self.render_text(f"x{self.combo} COMBO!", self.width // 2 - 150, 
                           100, combo_font, (255, 200, 50))
        
        # Статистика (верхний правый угол)
        stats_x = self.width - 300
        self.render_text(f"Hits: {self.hits}", stats_x, 30, self.small_font, (100, 255, 100))
        self.render_text(f"Miss: {self.missed}", stats_x, 80, self.small_font, (255, 100, 100))
        self.render_text(f"Max: x{self.max_combo}", stats_x, 130, self.small_font, (255, 255, 100))
        
        # Статус подключения
        if not self.connected:
            self.render_text("IMU NOT CONNECTED!", self.width // 2 - 200, 
                           self.height // 2 - 100, self.font, (255, 50, 50))
            self.render_text("Check XREAL glasses connection", self.width // 2 - 220,
                           self.height // 2, self.small_font, (255, 150, 150))
        
        # Инструкции (первые 5 секунд)
        if self.show_instructions:
            alpha = int(255 * min(1.0, self.instruction_timer))
            inst_y = self.height - 200
            self.render_text("SWING YOUR HANDS TO HIT CUBES!", 
                           self.width // 2 - 300, inst_y, 
                           self.small_font, (alpha, alpha, alpha))
            self.render_text("Red cubes = Left hand  |  Blue cubes = Right hand  |  Green = Any hand", 
                           self.width // 2 - 450, inst_y + 50, 
                           self.tiny_font, (alpha//2, alpha//2, alpha//2))
            self.render_text("Press ESC to quit", 
                           self.width // 2 - 150, inst_y + 100, 
                           self.tiny_font, (alpha//2, alpha//2, alpha//2))
        
        # Пауза
        if self.paused:
            self.render_text("PAUSED", self.width // 2 - 120, 
                           self.height // 2, self.font, (255, 255, 100))
            self.render_text("Press SPACE to continue", self.width // 2 - 180,
                           self.height // 2 + 60, self.small_font, (200, 200, 200))
        
        # Восстанавливаем 3D режим
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def render_text(self, text, x, y, font, color):
        """Рендер текста как OpenGL текстуры"""
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        glRasterPos2f(x, y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                     GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    def handle_events(self):
        """Обработка событий"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            
            elif event.type == KEYDOWN:
                # ESC для выхода
                if event.key == K_ESCAPE:
                    self.running = False
                    print("\nGame closed by user (ESC)")
                
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                    print("Game paused" if self.paused else "Game resumed")
                
                elif event.key == K_r:
                    self.reset_game()
    
    def reset_game(self):
        """Сброс игры"""
        self.cubes.clear()
        self.slashes.clear()
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.missed = 0
        self.hits = 0
        self.game_time = 0
        self.spawn_timer = 0
        self.spawn_interval = 1.2
        self.show_instructions = True
        self.instruction_timer = 5.0
        print("\n=== GAME RESET ===\n")
    
    def run(self):
        """Главный игровой цикл"""
        self.running = True
        clock = pygame.time.Clock()
        
        # Подключаемся к IMU
        print("\n" + "="*70)
        print(" "*20 + "XREAL BEAT SABER")
        print("="*70)
        print("\nConnecting to IMU sensor...")
        self.connect_imu()
        
        print("\nControls:")
        print("  🤜 SWING YOUR HANDS to hit cubes")
        print("  🔴 Red cubes = Left hand")
        print("  🔵 Blue cubes = Right hand")
        print("  🟢 Green cubes = Any hand")
        print("  ⌨️  ESC = Quit game")
        print("  ⌨️  SPACE = Pause")
        print("  ⌨️  R = Reset game")
        print("="*70 + "\n")
        
        last_time = time.time()
        
        while self.running:
            # Delta time
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)  # Ограничиваем dt
            last_time = current_time
            
            # События
            self.handle_events()
            
            # Обновление
            self.update(dt)
            
            # Отрисовка
            self.draw_3d()
            self.draw_hud()
            
            pygame.display.flip()
            clock.tick(60)
        
        # Очистка
        if self.imu_reader:
            self.imu_reader.stop()
        
        pygame.quit()
        
        # Финальная статистика
        print("\n" + "="*70)
        print(" "*25 + "GAME OVER")
        print("="*70)
        print(f"  Final Score:     {self.score:,}")
        print(f"  Max Combo:       x{self.max_combo}")
        print(f"  Total Hits:      {self.hits}")
        print(f"  Total Missed:    {self.missed}")
        if self.hits + self.missed > 0:
            accuracy = (self.hits / (self.hits + self.missed)) * 100
            print(f"  Accuracy:        {accuracy:.1f}%")
        print("="*70 + "\n")


def main():
    """Точка входа"""
    try:
        game = BeatSaberGame()
        game.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
