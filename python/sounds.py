import pygame

class Sounds:
    def __init__(self):
        self.hitmarkerSound = pygame.mixer.Sound(f'assets/sounds/hitmarker.mp3')
        self.damageSound = pygame.mixer.Sound(f'assets/sounds/minecraft-damage.mp3')
        self.deathSound = pygame.mixer.Sound(f'assets/sounds/minecraft-death.mp3')
        self.explosionSound = pygame.mixer.Sound(f'assets/sounds/minecraft-explosion.mp3')

    def playSound(self, sound):
        if sound == "hitmarker":
            self.hitmarkerSound.play()
        elif sound == "damage":
            self.damageSound.play()
        elif sound == "death":
            self.death.play()
        elif sound == "explosion":
            self.explosion.play()
