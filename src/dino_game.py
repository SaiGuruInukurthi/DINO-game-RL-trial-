"""Dino game module

Contains the Pygame-based Dino game implementation extracted from the
notebook. Provides Dinosaur, Obstacle, DinoGame classes and a small
CLI entrypoint for manual play.

Usage:
    python -m src.dino_game    # runs manual control mode

This file is intentionally self-contained so it can be imported by the
training notebook or used standalone.
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np

# Set SDL to use dummy video driver if DISPLAY is not available (for WSL/headless)
if not os.environ.get('DISPLAY'):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Physics
GRAVITY = 1.2
JUMP_STRENGTH = -18
GROUND_LEVEL = SCREEN_HEIGHT - 100

# Dino
DINO_WIDTH = 40
DINO_HEIGHT = 60
DINO_X = 100

# Obstacles
OBSTACLE_WIDTH = 20
OBSTACLE_HEIGHT = 50
OBSTACLE_SPEED = 8
MIN_OBSTACLE_GAP = 200
MAX_OBSTACLE_GAP = 400


class Dinosaur:
    """Dinosaur character that can jump and duck."""

    def __init__(self) -> None:
        self.x = DINO_X
        self.y = GROUND_LEVEL
        self.width = DINO_WIDTH
        self.height = DINO_HEIGHT
        self.velocity_y = 0.0
        self.is_jumping = False
        self.is_ducking = False

    def jump(self) -> None:
        if not self.is_jumping:
            self.velocity_y = JUMP_STRENGTH
            self.is_jumping = True

    def duck(self) -> None:
        if not self.is_jumping:
            self.is_ducking = True
            self.height = DINO_HEIGHT // 2

    def stand(self) -> None:
        self.is_ducking = False
        self.height = DINO_HEIGHT

    def update(self) -> None:
        if self.is_jumping:
            self.velocity_y += GRAVITY
            self.y += self.velocity_y
            if self.y >= GROUND_LEVEL:
                self.y = GROUND_LEVEL
                self.velocity_y = 0.0
                self.is_jumping = False

    def draw(self, screen: pygame.Surface) -> None:
        draw_y = int(self.y) if not self.is_ducking else int(self.y + DINO_HEIGHT // 2)
        pygame.draw.rect(screen, BLACK, (self.x, draw_y, self.width, self.height))

    def get_rect(self) -> pygame.Rect:
        draw_y = int(self.y) if not self.is_ducking else int(self.y + DINO_HEIGHT // 2)
        return pygame.Rect(self.x, draw_y, self.width, self.height)


class Obstacle:
    """Obstacle that moves leftwards toward the dinosaur."""

    def __init__(self, x: int, obstacle_type: str = "cactus") -> None:
        self.x = x
        self.type = obstacle_type
        if obstacle_type == "cactus":
            self.width = OBSTACLE_WIDTH
            self.height = OBSTACLE_HEIGHT
            self.y = GROUND_LEVEL
        elif obstacle_type == "bird":
            self.width = 40
            self.height = 30
            self.y = GROUND_LEVEL - random.choice([0, 50, 100])
        else:
            self.width = OBSTACLE_WIDTH
            self.height = OBSTACLE_HEIGHT
            self.y = GROUND_LEVEL

        self.speed = OBSTACLE_SPEED

    def update(self) -> None:
        self.x -= self.speed

    def draw(self, screen: pygame.Surface) -> None:
        if self.type == "cactus":
            pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height))
        else:
            pygame.draw.rect(screen, GRAY, (self.x, self.y, self.width, self.height))

    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def is_off_screen(self) -> bool:
        return self.x + self.width < 0


class DinoGame:
    """Encapsulates the game logic and rendering.

    This class is GUI-capable (uses Pygame) but does not depend on Gym.
    The training notebook may import this module and wrap it in a Gym
    interface if desired.
    """

    def __init__(self, render: bool = True) -> None:
        self.render_mode = render
        self.screen = None
        self.clock = None
        self.font = None
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Dino RL Game")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self) -> np.ndarray:
        self.dino = Dinosaur()
        self.obstacles: List[Obstacle] = []
        self.score = 0
        self.frames = 0
        self.game_over = False
        self.next_obstacle_frame = 60
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        closest = None
        min_dist = float("inf")
        for obs in self.obstacles:
            if obs.x > self.dino.x:
                dist = obs.x - self.dino.x
                if dist < min_dist:
                    min_dist = dist
                    closest = obs

        if closest:
            obs = np.array([
                self.dino.y / SCREEN_HEIGHT,
                self.dino.velocity_y / 20,
                min_dist / SCREEN_WIDTH,
                closest.height / SCREEN_HEIGHT,
                closest.y / SCREEN_HEIGHT,
                1 if self.dino.is_jumping else 0,
                1 if self.dino.is_ducking else 0,
                self.score / 1000,
            ], dtype=np.float32)
        else:
            obs = np.array([
                self.dino.y / SCREEN_HEIGHT,
                self.dino.velocity_y / 20,
                1.0,
                0.0,
                0.0,
                1 if self.dino.is_jumping else 0,
                1 if self.dino.is_ducking else 0,
                self.score / 1000,
            ], dtype=np.float32)
        return obs

    def _spawn_obstacle(self) -> None:
        obstacle_type = random.choice(["cactus", "cactus", "bird"])
        obstacle = Obstacle(SCREEN_WIDTH, obstacle_type)
        self.obstacles.append(obstacle)

    def _check_collision(self) -> bool:
        drect = self.dino.get_rect()
        return any(drect.colliderect(o.get_rect()) for o in self.obstacles)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.frames += 1

        # Actions: 0 = nothing, 1 = jump, 2 = duck
        if action == 1:
            self.dino.jump()
        elif action == 2:
            self.dino.duck()
        else:
            self.dino.stand()

        self.dino.update()
        for o in self.obstacles:
            o.update()

        initial = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if not o.is_off_screen()]
        if len(self.obstacles) < initial:
            self.score += 10

        if self.frames >= self.next_obstacle_frame:
            self._spawn_obstacle()
            gap = random.randint(MIN_OBSTACLE_GAP, MAX_OBSTACLE_GAP)
            self.next_obstacle_frame = self.frames + max(10, gap // OBSTACLE_SPEED)

        collision = self._check_collision()

        reward = 0.1
        if collision:
            reward = -100.0
            self.game_over = True
        elif len(self.obstacles) < initial:
            reward = 10.0

        obs = self._get_observation()
        done = self.game_over
        info = {"score": self.score, "frames": self.frames}
        return obs, reward, done, info

    def render(self) -> None:
        if not self.render_mode or self.screen is None:
            return
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, (0, GROUND_LEVEL + DINO_HEIGHT), (SCREEN_WIDTH, GROUND_LEVEL + DINO_HEIGHT), 2)
        self.dino.draw(self.screen)
        for o in self.obstacles:
            o.draw(self.screen)
        if self.font:
            text = self.font.render(f"Score: {self.score}", True, BLACK)
            self.screen.blit(text, (10, 10))
        pygame.display.flip()
        if self.clock:
            self.clock.tick(FPS)

    def close(self) -> None:
        if self.screen:
            pygame.quit()


def main_manual():
    """Run a manual control loop for testing/play."""
    game = DinoGame(render=True)
    obs = game.reset()
    done = False
    total_reward = 0.0

    try:
        while not done:
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
                action = 1
            elif keys[pygame.K_DOWN]:
                action = 2

            obs, reward, done, info = game.step(action)
            total_reward += reward
            game.render()

            if done:
                print("Game over")
                print(f"Score: {info['score']}  Frames: {info['frames']}")
                break
    finally:
        game.close()


if __name__ == "__main__":
    main_manual()
