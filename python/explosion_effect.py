import pygame
from particle import Particle
from utils import load_spritesheet
from constants import EXPLOSION_PATH

class ExplosionEffect:
    def __init__(self, ball_x, ball_y, ball_size, explosionSound=None):
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.ball_size = ball_size
        self.explosion_size_mult = 10
        self.explosion_speed = 6
        self.index = 0
        self.counter = 0
        self.finished = False
        self.active = True
        self.frames = load_spritesheet(EXPLOSION_PATH, 1, 12)
        self.frame = self.frames[self.index]
        self.explosion_size = int(
            self.ball_size * self.explosion_size_mult
        )  # Scale the explosion size
        self.blast_radius = self.explosion_size * 0.8
        if explosionSound is not None:
            explosionSound.play()

    def update(self):    
        self.counter += 1

        # Update the current frame if the counter reached the explosion speed
        if self.counter >= self.explosion_speed:
            self.counter = 0
            self.index += 1
            if self.index < len(self.frames):
                self.frame = self.frames[self.index]
            else:
                self.finished = True

    def draw(self, screen):
        if self.active and self.index < len(self.frames):
            sized_frame = pygame.transform.scale(
                self.frame, (self.explosion_size, self.explosion_size)
            )

            y_offset = int(
                self.explosion_size * 0.3
            )  # Offset, so the explosion happens right at the ball

            rectangle = sized_frame.get_rect(
                center=(self.ball_x, self.ball_y - y_offset)
            )  # Center coords of the explosion in the ball
            screen.blit(sized_frame, rectangle)  # Render frame
