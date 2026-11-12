import random

import pygame

from ash import Ash
from constants import ROOM_TEMP
from fire_effect import FireEffect


def make_circular_image(image_path, radius):
    size = radius * 2
    raw_image = pygame.image.load(image_path).convert_alpha()
    scaled_image = pygame.transform.scale(raw_image, (size, size))

    # Create a circular mask
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (radius, radius), radius)

    # Apply the circular mask to the scaled image
    circular_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    circular_surface.blit(scaled_image, (0, 0))
    circular_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return circular_surface


bounce_stop = 0.3  ##this variable is used to check for really tiny bounces and stop them if below the threshold
max_burn_level = (
    100.0  # This variable is maximum burn level before the ball is destroyed
)
ashes = []  # just like how the balls spawn

# I wanted to add some sort of liquid or something that could extinguish
#     the balls, but I'm not experienced in liquid simulations
# If someone after me could, that might be a cool idea! (or some other extinguish method)
# Setting fire_effect.active to false maintains the burn amount of this ball
#        so it works if the ball is extinguished in some way and then ignited again

# ^ I'm neither experienced, but I added temperature, which will decrease the heat if the
# ball is not burning yet. You can make the liquid (or whatever else) just decrease the
# temperature. A ball is currently burning above 400°C and also not loosing heat at that temp.


class Ball:
    def __init__(
        self,
        x_pos,
        y_pos,
        id,
        gravity,
        y_speed=0,
        x_speed=0,
        max_level=4,
        damageSound=None,
        deathSound=None,
    ):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = random.randint(
            25, 45
        )  ##every ball gets random attributes when its spawned
        # self.level = int(random.uniform(max(1, max_level-4),max_level))
        self.level = random.randint(1, max_level)
        self.image = make_circular_image(f"assets/balls/{self.level}.png", self.radius)
        self.density = random.uniform(0.5, 2)
        self.mass = int(
            20 + self.radius * 4 * self.density
        )  # 50 - 500, but based on size and random density to feel more natural
        self.retention = random.uniform(0.7, 0.95)
        self.friction = random.uniform(0.05, 0.3)
        self.angle = 0  # for making the balls rotate
        self.angular_speed = 0  # because apparently rotation is super complicated
        self.gravity = gravity
        self.y_speed = random.randint(-5, 5)
        self.x_speed = random.randint(-5, 5)
        self.id = id
        self.fire_effect = FireEffect(self.x_pos, self.y_pos, self.radius)
        self.temperature = ROOM_TEMP  # Temperature is in °C
        self.ignition_temp = 100.0  # Temperature required for a ball to catch on fire
        self.flame_temp = 400.0  # Temperature kept while flame is burning
        self.burn_level: float = 0.0
        self.active = True  # When this is set to false, the ball is removed from the simulation in main.py
        self.damageSound = damageSound
        self.deathSound = deathSound
        self.trail = []  # trail history array
        self.grabbed = False
        self.prevGrabbedPos = (self.x_pos, self.y_pos)
        self.scored = False
        self.shot_frames = 0

    def onClick(self):
        self.update_temp()
        self.grab()

    def draw(self, screen):
        # Glow on the objects based on temp
        heat_ratio = min(
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
        tex_rot = pygame.transform.rotate(texture, self.angle)

        # Fill the texture
        rect = tex_rot.get_rect(center=(int(self.x_pos), int(self.y_pos)))
        screen.blit(tex_rot, rect)
        # Draw the fire effect
        self.fire_effect.draw(screen)

    def get_center(self):
        return (self.x_pos, self.y_pos)

    def update_position(self, mouseX, mouseY):
        self.fire_effect.update_pos(self.x_pos, self.y_pos)
        self.fire_effect.update()
        # self.update_temp()
        self.update_fire()

        self.y_pos += self.y_speed
        self.x_pos += self.x_speed
        self.angle += self.angular_speed
        self.angular_speed *= 0.98

        if self.grabbed:
            self.handleGrab(mouseX, mouseY)

    def update_image(self):
        i = "basketball" if self.scored else self.level
        self.image = make_circular_image(f"assets/balls/{i}.png", self.radius)

    def apply_gravity_and_velocity(self, WINDOW_HEIGHT, WALL_THICKNESS):
        if (
            self.y_pos < self.floor(WINDOW_HEIGHT, WALL_THICKNESS)
        ):  ## if the ball's y position is less than the bottom of the screen pretty much
            self.y_speed += self.gravity

        if (
            self.y_speed == 0 and self.x_speed != 0
        ):  ## ball is rolling along the ground, slow it down
            if self.x_speed > 0:
                self.x_speed -= self.friction
            else:
                self.x_speed += self.friction
            if abs(self.x_speed) < self.friction:
                self.x_speed = 0

    def increase_burn(self):
        if self.fire_effect.active:
            self.burn_level = self.burn_level + random.uniform(0.1, 0.3)
        if self.burn_level >= max_burn_level:
            if self.deathSound is not None:
                self.deathSound.play()
            # ASHES nonsense here
            for dx in range(-int(self.radius), int(self.radius) + 1, 8):
                for dy in range(-int(self.radius), int(self.radius) + 1, 8):
                    if (
                        dx * dx + dy * dy <= self.radius * self.radius
                    ):  ##literally span then inside the entire ball circle
                        ashes.append(
                            Ash(
                                x_pos=self.x_pos + dx,
                                y_pos=self.y_pos + dy,
                                id=len(ashes),
                                gravity=self.gravity,
                                x_speed=self.x_speed
                                + random.uniform(
                                    -1, 1
                                ),  # give the ashes left/right motion
                                y_speed=self.y_speed + random.uniform(-1, 0),
                            )
                        )  # screw it, also up/down random
            self.active = False

    def floor(self, WINDOW_HEIGHT, WALL_THICKNESS):
        return WINDOW_HEIGHT - self.radius - (WALL_THICKNESS / 2)

    def ceiling(self, WINDOW_HEIGHT, WALL_THICKNESS):
        return 0 + self.radius + (WALL_THICKNESS / 2)

    def bounds(self, WINDOW_WIDTH, WINDOW_HEIGHT, WALL_THICKNESS):
        floor_y = self.floor(WINDOW_HEIGHT, WALL_THICKNESS)
        ceiling_y = self.ceiling(WINDOW_HEIGHT, WALL_THICKNESS)
        left_x = self.radius + WALL_THICKNESS / 2
        right_x = WINDOW_WIDTH - self.radius - WALL_THICKNESS / 2

        # ceiling
        if self.y_pos < ceiling_y:
            self.y_pos = ceiling_y

            self.angular_speed -= self.x_speed / self.radius

            rebound = self.y_speed * self.retention - self.gravity

            if rebound > bounce_stop:
                self.y_speed = -rebound
            else:
                self.y_speed = 0

        # floor
        if self.y_pos > floor_y:
            self.y_pos = floor_y

            self.angular_speed += self.x_speed / self.radius

            rebound = (
                self.y_speed * self.retention - self.gravity
            )  ## minus one frame of gravity, else the floor snap-up feeds balls energy and they bounce forever

            if rebound > bounce_stop:
                self.y_speed = -rebound
            else:
                self.y_speed = 0

        if self.x_pos < left_x:
            self.x_pos = left_x

            self.angular_speed -= self.x_speed / self.radius

            if self.x_speed < 0:
                self.x_speed *= -1 * self.retention

                if abs(self.x_speed) < bounce_stop:
                    self.x_speed = 0

        elif self.x_pos > right_x:
            self.x_pos = right_x

            self.angular_speed += self.x_speed / self.radius

            if self.x_speed > 0:
                self.x_speed *= -1 * self.retention

                if abs(self.x_speed) < bounce_stop:
                    self.x_speed = 0

    # I made this because the other one both checks if it's being selected
    #        and selects it, but I just wanted to check if it was within the click
    def check_if_selected(self, mouseX, mouseY):
        dx = mouseX - self.x_pos
        dy = mouseY - self.y_pos
        return dx**2 + dy**2 <= self.radius**2

    # Apply additional forces to the balls (for explosion)
    def apply_forces(self, force_x, force_y):
        self.x_speed += force_x
        self.y_speed += force_y

    def update_temp(self):
        # Cool down if the temp is above room temp
        if self.temperature > ROOM_TEMP:
            self.temperature -= 150
            self.temperature = max(ROOM_TEMP, self.temperature)

        if self.temperature >= self.ignition_temp:
            self.fire_effect.active = True
        else:
            self.temperature = ROOM_TEMP
            # self.burn_level = 0
            self.fire_effect.active = False

    def update_fire(self):
        if self.temperature > ROOM_TEMP:
            self.temperature -= 0.1
            self.temperature = max(ROOM_TEMP, self.temperature)

        if self.temperature >= self.ignition_temp:
            ## This will only play damage sound for one ball, it avoids having a bunch of burning sounds playing all at once
            if pygame.mixer.Channel(0).get_busy() == False:
                self.damageSound.play()
            self.fire_effect.active = True
        else:
            self.temperature = ROOM_TEMP
            # self.burn_level = 0
            self.fire_effect.active = False

    # this lets you click and hold the ball to throw it around!
    def grab(self):
        self.grabbed = True

    def releaseGrab(self):
        if not self.grabbed:
            return

        diffX = self.x_pos - self.prevGrabbedPos[0]
        diffY = self.y_pos - self.prevGrabbedPos[1]

        self.apply_forces(diffX, diffY)
        self.grabbed = False

        self.shot_frames = 120

    def handleGrab(self, mouseX, mouseY):
        self.prevGrabbedPos = (self.x_pos, self.y_pos)
        self.setPosToMouse(mouseX, mouseY)
        diffX = self.x_pos - self.prevGrabbedPos[0]
        diffY = self.y_pos - self.prevGrabbedPos[1]
        self.apply_forces(diffX, diffY)

    def setPosToMouse(self, mouseX, mouseY):
        self.x_pos = mouseX
        self.y_pos = mouseY
