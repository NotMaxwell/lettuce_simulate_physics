import sys
import random
import pygame

# I didn't do this, but I like the idea of gambling. Expanding the dice in some way would be awesome :)

# there's a die on screen that determines the gravity of the balls, higher roll, more gravity and vice versa

class Dice:
    def __init__(self, x, y):
        self.x_pos = x
        self.y_pos = y
        self.value = 1

        self.size = 70

        self.gravity_values = {  # higher roll, more gravity
            1: 0.6,
            2: 0.9,
            3: 1.2,
            4: 1.5,
            5: 1.8,
            6: 2,
        }
        self.font = pygame.font.SysFont(None, 40)

    def roll(self):
        self.value = random.randint(1, 6)
        return self.value

    def get_gravity(self):
        return self.gravity_values[self.value]

    def draw(self, screen):
        dice_rect = pygame.Rect(self.x_pos, self.y_pos, self.size, self.size)

        pygame.draw.rect(
            screen, "WHITE", dice_rect, border_radius=2
        )  # the white box is the die
        pygame.draw.rect(
            screen, "BLACK", dice_rect, width=2, border_radius=2
        )  # the black outline of the die
        text_surface = self.font.render(
            f"{self.value}", True, "BLACK"
        )  # the value of the die
        text_rect = text_surface.get_rect(
            center=dice_rect.center
        )  # centers the value of the die
        screen.blit(text_surface, text_rect)


if __name__ == "__main__":  ## only runs if the file is run directly to test and stuff
    pygame.init()

    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("dice test")

    clock = pygame.time.Clock()
    FPS = 60

    dice = Dice(100, 100)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                dice.roll()
                print("Roll:", dice.value)
                print("Gravity:", dice.get_gravity())

        screen.fill("BLACK")
        dice.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
