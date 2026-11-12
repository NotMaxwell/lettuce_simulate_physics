import pygame


# General particle class that can be used for any size, gravity, direction, etc
# Check out fire_effect.py for implementation of this class if you want to make
#   your own particle effect for something else
class Particle:
    def __init__(self, x_pos, y_pos, size, gravity, direction, time, color):
        self.x_pos = x_pos
        self.y_pos = y_pos
        # Particles are squares so only one size variable is used
        self.size = size
        # The gravity that this particle is affected by (can be negative)
        self.gravity = gravity
        # The amount this particle will move left or right
        self.direction = direction
        # Time to exist
        self.end_time = time
        self.time = 0
        self.color = color

    def draw(self, screen):
        particle_rect = pygame.Rect(self.x_pos, self.y_pos, self.size, self.size)
        pygame.draw.rect(screen, self.color, particle_rect)

    # Runs to update the gravity for this particle
    #   and increment time to delete it
    def update(self):
        self.time = self.time + 1
        self.y_pos += self.gravity
        self.x_pos += self.direction
