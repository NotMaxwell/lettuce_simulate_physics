import random
from particle import Particle


class FireEffect:
    def __init__(self, ball_x, ball_y, ball_size):
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.ball_size = ball_size
        self.particles = []  # Init empty particles list to be filled
        self.gravity = -3  # Hardcoded fire gravity
        self.particle_lifetime = 20
        self.active = False

    def start_particles(self):
        self.active = True

    def end_particles(self):
        self.active = False

    # Add particle to end of particle list
    def add_particle(self):
        # Random x and y positions based on the ball (can only spawn within the ball's region)
        particle_x = random.randint(
            int(self.ball_x) - int(self.ball_size),
            int(self.ball_x) + int(self.ball_size),
        )
        particle_y = random.randint(
            int(self.ball_y) - int(self.ball_size),
            int(self.ball_y) + int(self.ball_size),
        )
        particle_size = random.randint(2, 10)
        particle_direction = random.randint(-1, 1)
        particle_color = (255, random.randint(0, 140), 0)
        new_particle = Particle(
            particle_x,
            particle_y,
            particle_size,
            self.gravity,
            particle_direction,
            self.particle_lifetime,
            particle_color,
        )
        self.particles.append(new_particle)

    # Remove the selected particle from the list
    def remove_particle(self, particle):
        self.particles.remove(particle)
        # Add a new particle to the end of the list if the effect is still active
        if self.active:
            self.add_particle()

    # Update the ball position to the ball's position
    def update_pos(self, ball_x, ball_y):
        self.ball_x = ball_x
        self.ball_y = ball_y

    # Runs the particles in a fire-like animation
    def update(self):
        if len(self.particles) < 20 and self.active:
            self.add_particle()
        for particle in self.particles:
            particle.update()
            if particle.time > particle.end_time:
                self.remove_particle(particle)

    def draw(self, screen):
        for particle in self.particles:
            particle.draw(screen)
