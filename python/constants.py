import os

# Paths
EXPLOSION_PATH = f"assets/explosion/explosion.png"
BALLS_DIR = os.path.join(os.path.dirname(__file__), "assets/balls")


# Pygame
WINDOW_WIDTH, WINDOW_HEIGHT = 1029, 720  ##70mm IMAX aspect ratio : )
BACKGROUND_COLOR = "blue4"
FPS = 60


# Misc
LASSO_FADE_TIME = 1000  # ms
WALL_THICKNESS = 10
BASE_FORCE = 10_000  # Force for explosion on other balls
MAX_TRAIL = 30       # Ball trail max length


# Heat
ROOM_TEMP = 21.0  # Room temperature in °C
BASE_CONDUCT_MULT = 0.0005  # Conductivity multiplier for fire spread
IMPACT_HEAT_MULT = 1.5  # Heat multiplication for kinetic impacts
BASE_EXPLOSION_HEAT = 350  # Heat of the explosion
