import os
import pygame
import random
from fire_effect import FireEffect
from explosion_effect import ExplosionEffect
from utils import calc_mouse_vector
from constants import ROOM_TEMP, BALLS_DIR

##### I just copied the ball object, so there's a bunch of junk in here

bounce_stop = 0.3  ##this variable is used to check for really tiny bounces and stop them if below the threshold
max_burn_level = (
    100.0  # This variable is maximum burn level before the ball is destroyed
)

# I wanted to add some sort of liquid or something that could extinguish
#     the balls, but I'm not experienced in liquid simulations
# If someone after me could, that might be a cool idea! (or some other extinguish method)
# Setting fire_effect.active to false maintains the burn amount of this ball
#        so it works if the ball is extinguished in some way and then ignited again

# ^ I'm neither experienced, but I added temperature, which will decrease the heat if the 
# ball is not burning yet. You can make the liquid (or whatever else) just decrease the
# temperature. A ball is currently burning above 400°C and also not loosing heat at that temp.


class Ash:
    def __init__(self, x_pos, y_pos, id, gravity, y_speed=0, x_speed=0):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = 2
        #self.image = #make_circular_image(random.choice(sprite_paths), self.radius)
        self.start_color = random.randint(1, 255)
        self.color = (self.start_color, 0, 0)
        self.spawn_time = pygame.time.get_ticks()
        self.lifetime = 3000  # milliseconds = 6 seconds
        self.density = random.uniform(0.5, 2)
        self.mass = int(
            20 + self.radius * 4 * self.density
        )  # 50 - 500, but based on size and random density to feel more natural
        self.retention = random.uniform(0.7, 0.95)
        self.friction = random.uniform(0.05, 0.3)
        self.gravity = random.uniform(0.1, 1)
        self.y_speed = y_speed
        self.x_speed = x_speed
        self.id = id
        self.fire_effect = FireEffect(self.x_pos, self.y_pos, self.radius)
        self.temperature = ROOM_TEMP  # Temperature is in °C
        self.ignition_temp = 100.0  # Temperature required for a ball to catch on fire
        self.flame_temp = 400.0  # Temperature kept while flame is burning
        self.burn_level: float = 0.0
        self.active = True  # When this is set to false, the ball is removed from the simulation in main.py

    def draw(self, screen):
        # Glow on the objects based on temp
        pygame.draw.circle(screen, self.color, (int(self.x_pos), int(self.y_pos)), self.radius)
        '''heat_ratio = min(
            1.0,
            max(0.0, self.temperature - ROOM_TEMP) / (self.ignition_temp - ROOM_TEMP),
        )

        # Make the image to coal
        burn_ratio = min(1.0, self.burn_level / max_burn_level)
        char_factor = 1.0 - burn_ratio

        # Combine the glow with coal effect
        red_color = int(255 * char_factor)
        green_blue_color = int(255 * (1.0 - heat_ratio * 0.5) * char_factor)

        texture = self.image.copy()
        texture.fill(
            (red_color, green_blue_color, green_blue_color, 255),
            special_flags=pygame.BLEND_RGBA_MULT,
        )

        # Fill the texture
        rect = texture.get_rect(center=(int(self.x_pos), int(self.y_pos)))
        screen.blit(texture, rect)
        # Draw the fire effect
        self.fire_effect.draw(screen)'''

    def get_center(self):
        return (self.x_pos, self.y_pos)

    def update_position(self):
        self.fire_effect.update_pos(self.x_pos, self.y_pos)
        self.fire_effect.update()
        self.update_temp()

        self.y_pos += self.y_speed
        self.x_pos += self.x_speed

    def apply_gravity_and_velocity(self, WINDOW_HEIGHT, WALL_THICKNESS):
        if self.y_pos < self.floor(WINDOW_HEIGHT, WALL_THICKNESS):  ## if the ball's y position is less than the bottom of the screen pretty much
            self.y_speed += self.gravity

        if (self.y_speed == 0 and self.x_speed != 0):  ## ball is rolling along the ground, slow it down
            if self.x_speed > 0:
                self.x_speed -= self.friction
            else:
                self.x_speed += self.friction
            if abs(self.x_speed) < self.friction:
                self.x_speed = 0

    def increase_burn(self):
        life = pygame.time.get_ticks() - self.spawn_time
        burn = min(life / self.lifetime, 1.0)
        self.color = (int(self.start_color * (1.0 - burn)), 0, 0)
        if life >= self.lifetime:
            self.active = False

    def floor(self, WINDOW_HEIGHT, WALL_THICKNESS):
        return WINDOW_HEIGHT - self.radius - (WALL_THICKNESS / 2)

    def bounds(self, WINDOW_WIDTH, WINDOW_HEIGHT, WALL_THICKNESS):
        floor_y = self.floor(WINDOW_HEIGHT, WALL_THICKNESS)
        left_x = self.radius + WALL_THICKNESS / 2
        right_x = WINDOW_WIDTH - self.radius - WALL_THICKNESS / 2

        if self.y_pos > floor_y:
            self.y_pos = floor_y

            rebound = (
                self.y_speed * self.retention - self.gravity
            )  ## minus one frame of gravity, else the floor snap-up feeds balls energy and they bounce forever
            if rebound > bounce_stop:
                self.y_speed = -rebound
            else:
                self.y_speed = 0

        if self.x_pos < left_x:
            self.x_pos = left_x

            if self.x_speed < 0:
                self.x_speed *= -1 * self.retention

                if abs(self.x_speed) < bounce_stop:
                    self.x_speed = 0

        elif self.x_pos > right_x:
            self.x_pos = right_x

            if self.x_speed > 0:
                self.x_speed *= -1 * self.retention

                if abs(self.x_speed) < bounce_stop:
                    self.x_speed = 0

    # Apply additional forces to the balls (for explosion)
    def apply_forces(self, force_x, force_y):
        self.x_speed += force_x
        self.y_speed += force_y

    def update_temp(self):
        # Keep burning balls at high temp
        if self.fire_effect.active:
            self.temperature = self.flame_temp
        else:
            # Cool down if the temp is above room temp
            if self.temperature > ROOM_TEMP:
                self.temperature -= 0.15
                self.temperature = max(ROOM_TEMP, self.temperature)

            if self.temperature >= self.ignition_temp:
                self.fire_effect.active = True
