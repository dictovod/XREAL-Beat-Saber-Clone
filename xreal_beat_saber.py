#!/usr/bin/env python3
"""
XREAL Beat Saber Clone - Full Version
Ритм-игра с распознаванием рук через камеру XREAL

Управление:
- Махи руками для отбивания кубиков (детекция через камеру)
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

from hand_tracker import XrealHandTracker, Hand
from config import GLASSES_IP_PRIMARY, PORT_IMU


class Cube:
    """Летящий кубик"""
    def __init__(self, lane, color, spawn_time, required_hand):
        self.lane = lane  # -1, 0, 1 (левый, центр, правый)
        self.color = color  # (r, g, b)
        self.z = -50  # Начальная позиция вдали
        self.spawn_time = spawn_time
        self.required_hand = required_hand  # 'left', 'right', 'both'
        self.hit = False
        self.size = 1.0
        self.hit_time = 0
        
    def update(self, dt):
        """Обновить позицию"""
        self.z += 20 * dt  # Скорость приближения
        
        # Анимация попадания
        if self.hit:
            age = time.time() - self.hit_time
            self.size = 1.0 + age * 2
    
    def is_in_hit_zone(self):
        """Проверка в зоне удара"""
        return -3 < self.z < 1
    
    def is_missed(self):
        """Проверка промаха"""
        return self.z > 5
    
    def get_world_position(self):
        """Получить мировые координаты кубика"""
        x = self.lane * 3.5
        return np.array([x, 0, self.z])
    
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
        
        # Рёбра
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        
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


class BeatSaberGame:
    """Основной класс игры с распознаванием рук"""
    
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
        pygame.mouse.set_visible(False)
        
        # OpenGL настройки
        self.setup_opengl()
        
        # Hand Tracker
        self.hand_tracker = XrealHandTracker()
        self.tracker_active = False
        
        # Игровая логика
        self.cubes = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.missed = 0
        self.hits = 0
        self.game_time = 0
        self.spawn_timer = 0
        self.spawn_interval = 1.2
        
        # Отслеживание ударов
        self.last_left_hit = 0
        self.last_right_hit = 0
        self.hit_cooldown = 0.3
        
        # Визуализация рук в 3D
        self.hand_trails = {'left': [], 'right': []}
        self.max_trail_length = 20
        
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
        self.paused = False
        self.show_instructions = True
        self.instruction_timer = 10.0
        
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
        
        glClearColor(0.05, 0.05, 0.15, 1.0)
    
    def hand_to_world_coords(self, hand: Hand) -> np.ndarray:
        """
        Конвертировать координаты руки (0-1) в мировые координаты игры
        """
        # Запястье руки
        wrist = hand.wrist_pos
        
        # Маппинг координат камеры -> мир игры
        # X: 0-1 -> -10 до +10 (ширина игрового поля)
        # Y: 0-1 -> +5 до -5 (высота, инвертирована)
        # Z: фиксированная позиция в зоне удара (0)
        
        world_x = (wrist[0] - 0.5) * 20  # Центрируем и масштабируем
        world_y = (0.5 - wrist[1]) * 10  # Инвертируем Y
        world_z = 0  # Руки всегда в плоскости удара
        
        return np.array([world_x, world_y, world_z])
    
    def detect_hand_strikes(self) -> tuple:
        """
        Определить удары руками по скорости движения
        Возвращает (left_strike, right_strike, left_pos, right_pos)
        """
        left_hand, right_hand = self.hand_tracker.get_hands()
        
        left_strike = False
        right_strike = False
        left_pos = None
        right_pos = None
        
        current_time = time.time()
        
        # Левая рука
        if left_hand:
            left_pos = self.hand_to_world_coords(left_hand)
            
            # Добавляем в след
            self.hand_trails['left'].append((left_pos.copy(), current_time))
            if len(self.hand_trails['left']) > self.max_trail_length:
                self.hand_trails['left'].pop(0)
            
            # Проверяем скорость (только X и Y, игнорируем Z)
            speed = np.linalg.norm(left_hand.velocity[:2])
            
            if speed > 2.0 and current_time - self.last_left_hit > self.hit_cooldown:
                left_strike = True
                self.last_left_hit = current_time
                print(f"LEFT HAND STRIKE! Speed: {speed:.2f}")
        else:
            self.hand_trails['left'].clear()
        
        # Правая рука
        if right_hand:
            right_pos = self.hand_to_world_coords(right_hand)
            
            # Добавляем в след
            self.hand_trails['right'].append((right_pos.copy(), current_time))
            if len(self.hand_trails['right']) > self.max_trail_length:
                self.hand_trails['right'].pop(0)
            
            # Проверяем скорость
            speed = np.linalg.norm(right_hand.velocity[:2])
            
            if speed > 2.0 and current_time - self.last_right_hit > self.hit_cooldown:
                right_strike = True
                self.last_right_hit = current_time
                print(f"RIGHT HAND STRIKE! Speed: {speed:.2f}")
        else:
            self.hand_trails['right'].clear()
        
        return left_strike, right_strike, left_pos, right_pos
    
    def check_collision(self, hand_pos: np.ndarray, cube: Cube) -> bool:
        """Проверить столкновение руки с кубиком"""
        if hand_pos is None:
            return False
        
        cube_pos = cube.get_world_position()
        distance = np.linalg.norm(hand_pos - cube_pos)
        
        # Радиус столкновения
        hit_radius = 2.5
        
        return distance < hit_radius
    
    def spawn_cube(self):
        """Создать новый кубик"""
        lane = random.choice([-1, 0, 1])
        
        color_choices = [
            ((1, 0.2, 0.2), 'left'),    # Красный - левая
            ((0.2, 0.4, 1), 'right'),   # Синий - правая
            ((0.2, 1, 0.2), 'both'),    # Зелёный - любая
        ]
        color, hand = random.choice(color_choices)
        
        cube = Cube(lane, color, self.game_time, hand)
        self.cubes.append(cube)
    
    def check_hits(self, left_strike, right_strike, left_pos, right_pos):
        """Проверить попадания по кубикам"""
        for cube in self.cubes:
            if cube.hit or not cube.is_in_hit_zone():
                continue
            
            hit = False
            
            # Левая рука
            if left_strike and left_pos is not None:
                if cube.required_hand in ['left', 'both']:
                    if self.check_collision(left_pos, cube):
                        hit = True
            
            # Правая рука
            if right_strike and right_pos is not None:
                if cube.required_hand in ['right', 'both']:
                    if self.check_collision(right_pos, cube):
                        hit = True
            
            if hit:
                cube.hit = True
                cube.hit_time = time.time()
                self.score += 100 * (self.combo + 1)
                self.combo += 1
                self.max_combo = max(self.max_combo, self.combo)
                self.hits += 1
                print(f"HIT! Score: {self.score}, Combo: x{self.combo}")
                break
    
    def update(self, dt):
        """Обновить игровое состояние"""
        if self.paused:
            return
        
        self.game_time += dt
        
        # Обновляем hand tracker
        if self.tracker_active:
            self.hand_tracker.update()
        
        # Инструкции
        if self.show_instructions:
            self.instruction_timer -= dt
            if self.instruction_timer <= 0:
                self.show_instructions = False
        
        # Определить удары
        left_strike, right_strike, left_pos, right_pos = self.detect_hand_strikes()
        
        # Спавн кубиков
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self.spawn_cube()
            self.spawn_interval = max(0.6, 1.2 - self.game_time * 0.015)
        
        # Обновить кубики
        for cube in self.cubes[:]:
            cube.update(dt)
            
            if cube.is_missed():
                self.cubes.remove(cube)
                if not cube.hit:
                    self.missed += 1
                    self.combo = 0
                    print(f"MISS! Total missed: {self.missed}")
            
            if cube.hit and (time.time() - cube.hit_time) > 0.5:
                self.cubes.remove(cube)
        
        # Проверить попадания
        self.check_hits(left_strike, right_strike, left_pos, right_pos)
    
    def draw_3d(self):
        """Отрисовка 3D сцены"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Камера
        gluLookAt(
            0, 3, 8,
            0, 0, -10,
            0, 1, 0
        )
        
        # Сцена
        self.draw_floor()
        self.draw_lanes()
        
        # Кубики
        for cube in self.cubes:
            cube.draw()
        
        # Руки в 3D
        self.draw_hands_3d()
        
        # Зона удара
        self.draw_hit_zone()
    
    def draw_hands_3d(self):
        """Отрисовать руки и их следы в 3D"""
        glDisable(GL_LIGHTING)
        current_time = time.time()
        
        # Левая рука (красная)
        if self.hand_trails['left']:
            glLineWidth(4)
            glBegin(GL_LINE_STRIP)
            for i, (pos, t) in enumerate(self.hand_trails['left']):
                age = current_time - t
                alpha = max(0, 1 - age * 2)
                glColor4f(1, 0.3, 0.3, alpha)
                glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
            
            # Текущая позиция (сфера)
            last_pos = self.hand_trails['left'][-1][0]
            glColor4f(1, 0, 0, 0.8)
            glPushMatrix()
            glTranslatef(last_pos[0], last_pos[1], last_pos[2])
            self.draw_sphere(0.3, 10, 10)
            glPopMatrix()
        
        # Правая рука (синяя)
        if self.hand_trails['right']:
            glLineWidth(4)
            glBegin(GL_LINE_STRIP)
            for i, (pos, t) in enumerate(self.hand_trails['right']):
                age = current_time - t
                alpha = max(0, 1 - age * 2)
                glColor4f(0.3, 0.3, 1, alpha)
                glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
            
            # Текущая позиция (сфера)
            last_pos = self.hand_trails['right'][-1][0]
            glColor4f(0, 0, 1, 0.8)
            glPushMatrix()
            glTranslatef(last_pos[0], last_pos[1], last_pos[2])
            self.draw_sphere(0.3, 10, 10)
            glPopMatrix()
        
        glEnable(GL_LIGHTING)
    
    def draw_sphere(self, radius, slices, stacks):
        """Нарисовать сферу (упрощённая)"""
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = radius * math.sin(lat0)
            zr0 = radius * math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = radius * math.sin(lat1)
            zr1 = radius * math.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                glVertex3f(x * zr0, y * zr0, z0)
                glVertex3f(x * zr1, y * zr1, z1)
            glEnd()
    
    def draw_floor(self):
        """Рисовать пол с сеткой"""
        glDisable(GL_LIGHTING)
        glColor3f(0.1, 0.1, 0.2)
        glLineWidth(1)
        
        glBegin(GL_LINES)
        for i in range(-25, 10, 2):
            glVertex3f(-15, -3, i)
            glVertex3f(15, -3, i)
        
        for i in range(-7, 8, 2):
            glVertex3f(i*2, -3, -25)
            glVertex3f(i*2, -3, 10)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_lanes(self):
        """Рисовать дорожки"""
        glDisable(GL_LIGHTING)
        
        for lane in [-1, 0, 1]:
            if lane == -1:
                glColor4f(1, 0.2, 0.2, 0.15)
            elif lane == 1:
                glColor4f(0.2, 0.4, 1, 0.15)
            else:
                glColor4f(0.2, 1, 0.2, 0.15)
            
            glBegin(GL_QUADS)
            x = lane * 3.5
            glVertex3f(x - 1.5, -2.99, -50)
            glVertex3f(x + 1.5, -2.99, -50)
            glVertex3f(x + 1.5, -2.99, 10)
            glVertex3f(x - 1.5, -2.99, 10)
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_hit_zone(self):
        """Рисовать зону удара"""
        glDisable(GL_LIGHTING)
        glColor4f(1, 1, 0, 0.3)
        glLineWidth(3)
        
        glBegin(GL_LINES)
        glVertex3f(-12, -2.9, 0)
        glVertex3f(12, -2.9, 0)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_hud(self):
        """Рисовать HUD"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Счёт
        self.render_text(f"SCORE: {self.score}", 30, 30, self.font, (255, 255, 255))
        
        # Комбо
        if self.combo > 0:
            combo_size = min(100, 72 + self.combo * 2)
            combo_font = pygame.font.Font(None, combo_size)
            self.render_text(f"x{self.combo} COMBO!", self.width // 2 - 150, 
                           100, combo_font, (255, 200, 50))
        
        # Статистика
        stats_x = self.width - 300
        self.render_text(f"Hits: {self.hits}", stats_x, 30, self.small_font, (100, 255, 100))
        self.render_text(f"Miss: {self.missed}", stats_x, 80, self.small_font, (255, 100, 100))
        self.render_text(f"Max: x{self.max_combo}", stats_x, 130, self.small_font, (255, 255, 100))
        
        # Статус камеры
        if not self.tracker_active:
            self.render_text("CAMERA NOT CONNECTED!", self.width // 2 - 250, 
                           self.height // 2 - 100, self.font, (255, 50, 50))
            self.render_text("Check XREAL glasses connection", self.width // 2 - 220,
                           self.height // 2, self.small_font, (255, 150, 150))
        
        # Инструкции
        if self.show_instructions:
            alpha = int(255 * min(1.0, self.instruction_timer / 10.0))
            inst_y = self.height - 250
            
            self.render_text("WAVE YOUR HANDS IN FRONT OF CAMERA!", 
                           self.width // 2 - 350, inst_y, 
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
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def render_text(self, text, x, y, font, color):
        """Рендер текста"""
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
                if event.key == K_ESCAPE:
                    self.running = False
                    print("\nGame closed by user (ESC)")
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                elif event.key == K_r:
                    self.reset_game()
    
    def reset_game(self):
        """Сброс игры"""
        self.cubes.clear()
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.missed = 0
        self.hits = 0
        self.game_time = 0
        self.spawn_timer = 0
        self.spawn_interval = 1.2
        self.show_instructions = True
        self.instruction_timer = 10.0
        self.hand_trails = {'left': [], 'right': []}
        print("\n=== GAME RESET ===\n")
    
    def run(self):
        """Главный игровой цикл"""
        self.running = True
        clock = pygame.time.Clock()
        
        print("\n" + "="*70)
        print(" "*20 + "XREAL BEAT SABER")
        print("="*70)
        print("\nStarting hand tracking...")
        
        # Запускаем hand tracker
        if self.hand_tracker.start():
            self.tracker_active = True
            print("✓ Hand tracking active!")
        else:
            print("✗ Hand tracking failed - check camera connection")
            print("  Game will continue but without hand detection")
        
        print("\nControls:")
        print("  👋 WAVE YOUR HANDS in front of camera to hit cubes")
        print("  🔴 Red cubes = Left hand")
        print("  🔵 Blue cubes = Right hand")
        print("  🟢 Green cubes = Any hand")
        print("  ⌨️  ESC = Quit game")
        print("  ⌨️  SPACE = Pause")
        print("  ⌨️  R = Reset game")
        print("="*70 + "\n")
        
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)
            last_time = current_time
            
            self.handle_events()
            self.update(dt)
            self.draw_3d()
            self.draw_hud()
            
            pygame.display.flip()
            clock.tick(60)
        
        # Очистка
        if self.tracker_active:
            self.hand_tracker.stop()
        
        pygame.quit()
        
        # Статистика
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
