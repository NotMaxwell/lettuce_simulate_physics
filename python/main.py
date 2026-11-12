import math
import random
import sys

import pygame

from ball import Ball, ashes
from constants import *
from dice import Dice
from explosion_effect import ExplosionEffect
from utils import *

# todo: DELETE ME!!
"""
1. balls are combining into each other right away (maybe increase level variance)
2. balls are spawning too slowly (spawn multiple balls per frame when ballsPerClick is high)


1. dragging a ball into another ball does not apply the correct force
"""


## This is all a pretty standard pygame setup. Initalizing the window and a game clock.

pygame.init()
pygame.mixer.init()

## Init sounds
hitmarkerSound = pygame.mixer.Sound(f"assets/sounds/hitmarker.mp3")
damageSound = pygame.mixer.Sound(f"assets/sounds/minecraft-damage.mp3")
deathSound = pygame.mixer.Sound(f"assets/sounds/minecraft-death.mp3")
explosionSound = pygame.mixer.Sound(f"assets/sounds/minecraft-explosion.mp3")

backgroundMusic = pygame.mixer.music.load(f"assets/sounds/fortnite-corail.mp3")
# backgroundMusic = pygame.mixer.music.load(f"assets/sounds/monkey.mp3")
pygame.mixer.music.play(-1)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SCALED, vsync=1)
pygame.display.set_caption("FireBall Simulation")

clock = pygame.time.Clock()

keyRepeatDelay = 500
keyRepeatInterval = 80
pygame.key.set_repeat(keyRepeatDelay, keyRepeatInterval)

font = pygame.font.SysFont(None, 24)
help_text = font.render("Press [LEFT CLICK] to spawn ball.", True, "WHITE")
help_text2 = font.render("Press [RIGHT CLICK] to ignite a ball.", True, "WHITE")
help_text3 = font.render(
    "Hold [SHIFT] while dragging mouse to lasso. Lasso around ball to make boom boom.",
    True,
    "WHITE",
)
help_text4 = font.render(
    "Press [M] to mute background music.",
    True,
    "WHITE",
)
help_text5 = font.render(
    "Hold [G + LEFT CLICK] and drag to attract balls to your mouse.",
    True,
    "WHITE",
)
gravity_text = font.render("Press [F] to toggle gravity.", True, "WHITE")

numFramesToSpawnBalls = 0
ballsToSpawnPerFrame = 0
exactRemainingBalls = 0
ballsPerClick = 1
ballsPerClickFont = pygame.font.SysFont(None, 64, bold=True)


def updateBallsPerClickHelpText():
    global ballsPerClickHelpText
    ballsPerClickString = f"balls per click: {ballsPerClick}"
    ballsPerClickHelpText = ballsPerClickFont.render(ballsPerClickString, True, "GREEN")


updateBallsPerClickHelpText()
ballsPerClickPlusMinusFont = pygame.font.SysFont(None, 32, bold=True)
ballsPerClickPlusMinusHelpText = ballsPerClickPlusMinusFont.render(
    "(use + or -)", True, "GREEN"
)

big_font = pygame.font.SysFont(None, 32, bold=True)

##GAME VARIABLES
gravity = 0.6
mouse_trajectory = []

balls = []  ##starts empty, click an empty spot to spawn a ball
explosions = []
dice = Dice(900, 25)
current_level = 4
won = False
gravity_enabled = True

HOOP_LEFT = 335
HOOP_RIGHT = 375
HOOP_Y = 450


def draw_max_balls():
    pygame.draw.rect(screen, (20, 20, 20), (880, 15, 135, 105), border_radius=12)
    pygame.draw.rect(screen, (100, 100, 100), (880, 15, 135, 105), 2, border_radius=12)

    if not balls:
        return
    global current_level

    max_ball = max(balls, key=lambda ball: ball.level)
    current_level = max(4, max_ball.level)
    image = pygame.transform.scale(max_ball.image, (80, 80))
    screen.blit(image, (900, 25))

    text = big_font.render(str(max_ball.level), True, "WHITE")
    screen.blit(text, (990, 50))


def draw_attraction_bar():
    bar_x, bar_y, bar_width, bar_height = 880, 130, 135, 20

    pygame.draw.rect(
        screen, (20, 20, 20), (bar_x, bar_y, bar_width, bar_height), border_radius=6
    )

    fill_width = int(bar_width * (attraction_energy / 75))
    if fill_width > 0:
        pygame.draw.rect(
            screen,
            (40, 120, 255),
            (bar_x, bar_y, fill_width, bar_height),
            border_radius=6,
        )

    pygame.draw.rect(
        screen,
        (100, 100, 100),
        (bar_x, bar_y, bar_width, bar_height),
        2,
        border_radius=6,
    )


# Calculating explosion forces
def explosion_knockback(explosion_x, explosion_y, balls, blast_radius):
    for ball in balls:
        if not ball.active:
            continue

        # Direction vector
        dx = ball.x_pos - explosion_x
        dy = ball.y_pos - explosion_y

        # Distance
        d = math.sqrt(dx**2 + dy**2)

        # Unit vector for the direction
        if 0 < d <= blast_radius:
            dir_x = dx / d
            dir_y = dy / d

            # Calculate the applied force based on mass, distance, blast radius, base force & direction
            force = (BASE_FORCE / (ball.mass)) * (1 - d / blast_radius)
            force_x = dir_x * force
            force_y = dir_y * force

            # Apply forces and fire effect upon explosion
            ball.apply_forces(force_x, force_y)

            # Heat falloff over the distance
            falloff = 1 - (d / blast_radius)

            transfered_heat = (BASE_EXPLOSION_HEAT / (ball.mass / 100.0)) * falloff
            ball.temperature += transfered_heat


lasso_active = False
lasso_points = []
lasso_fade_start = None

##MAIN GAME LOOP (runs 4ever)
running = True
mute = False
countdown_counter, countdown_text = 120, "120".rjust(3)
pygame.time.set_timer(pygame.USEREVENT, 1000)

attraction_energy = 75


def activate_attraction():
    return bool(
        pygame.mouse.get_pressed()[0]
        and pygame.key.get_pressed()[pygame.K_g]
        and attraction_energy > 0
    )


def onLeftClick(mouseX, mouseY):

    global ballsToSpawnPerFrame
    global numFramesToSpawnBalls
    global exactRemainingBalls

    for ball in balls:
        if ball.check_if_selected(
            mouseX, mouseY
        ):  # pass the position of the mouse over to the function
            ball.onClick()
            return

    # WARNING: Don't try to understand this part. It doesn't make any sense!!

    # still spawning current batch of balls
    if numFramesToSpawnBalls > 0:
        return

    ballSpawnRatio = 1 / 3

    ballsToSpawn = ballsPerClick
    numFramesToSpawnBalls = int(ballsToSpawn ** (ballSpawnRatio))
    ballsToSpawnPerFrame = int(ballsToSpawn / numFramesToSpawnBalls)

    roundedTotal = numFramesToSpawnBalls * ballsToSpawnPerFrame

    exactRemainingBalls = ballsToSpawn - roundedTotal


def onLeftClickUp():

    # Sorry
    for ball in balls:
        ball.releaseGrab()


def onRightClick(mouseX, mouseY):
    for ball in balls:
        if ball.check_if_selected(mouseX, mouseY):  # If ball clicked
            ball.temperature = 400
            ball.update_temp()


def onPlusKeyPress():
    global ballsPerClick

    # unlimited cap on number of balls. DO NOT CHANGE <<<<< CRITICAL
    ballsPerClick += 1
    updateBallsPerClickHelpText()


def onMinusKeyPress():
    global ballsPerClick
    if ballsPerClick > 1:
        ballsPerClick -= 1
        updateBallsPerClickHelpText()


def createBall(mouseX, mouseY):
    dice.roll()
    ball_gravity = dice.get_gravity()
    balls.append(
        Ball(
            x_pos=mouseX,
            y_pos=mouseY,
            id=len(balls),
            gravity=ball_gravity,
            max_level=current_level,
            damageSound=damageSound,
            deathSound=deathSound,
        )
    )


while running:
    ##EVENT HANDLING
    for event in pygame.event.get():
        if (
            event.type == pygame.QUIT
        ):  ##basically if the user closes the window, stop running da game
            running = False

        mouseX, mouseY = pygame.mouse.get_pos()

        if numFramesToSpawnBalls > 0:
            extraBalls = 0
            if numFramesToSpawnBalls == 1:
                extraBalls = exactRemainingBalls

            for i in range(ballsToSpawnPerFrame + extraBalls):
                createBall(mouseX, mouseY)

            numFramesToSpawnBalls -= 1

        if event.type == pygame.USEREVENT:
            if not won:
                countdown_counter -= 1
                countdown_text = (
                    str(countdown_counter).rjust(3)
                    if countdown_counter > 0
                    else "You Lose!!"
                )

            # Use the same one second timer for attraction energy
            if activate_attraction():
                attraction_energy -= 25
            elif attraction_energy < 75:
                attraction_energy += 10

        if event.type == pygame.KEYDOWN and event.key == pygame.K_m and mute == False:
            pygame.mixer.music.set_volume(0.0)
            mute = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_m and mute == True:
            pygame.mixer.music.set_volume(1.0)
            mute = False

        # Let user start drawing lasso
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                lasso_active = True
                lasso_points = [event.pos]

        if event.type == pygame.MOUSEMOTION and lasso_active:
            lasso_points.append(event.pos)

        if activate_attraction():
            for ball in balls:
                tmp_ball_x = mouseX - ball.x_pos
                tmp_ball_y = mouseY - ball.y_pos
                tmp_ball_x *= 0.1
                tmp_ball_y *= 0.1

                ball.x_speed = 0
                ball.y_speed = 0
                ball.angular_speed = 0
                ball.apply_forces(tmp_ball_x, tmp_ball_y)

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            onLeftClickUp()

            if lasso_active:
                lasso_active = False
                lasso_fade_start = pygame.time.get_ticks()

                # Check which balls are in lasso area
                if len(lasso_points) >= 3:
                    for ball in balls:
                        if point_in_polygon(ball.get_center(), lasso_points):
                            # Cause explosion if the ball gets lassoed
                            explosion = ExplosionEffect(
                                ball.x_pos, ball.y_pos, ball.radius, explosionSound
                            )
                            explosions.append(explosion)

                            explosion_knockback(
                                ball.x_pos, ball.y_pos, balls, explosion.blast_radius
                            )

                            ball.active = False

        # # PLUS KEY
        if event.type == pygame.KEYDOWN and (
            event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS
        ):
            onPlusKeyPress()

        # # MINUS KEY
        if event.type == pygame.KEYDOWN and event.key == pygame.K_MINUS:
            onMinusKeyPress()

        # # LEFT CLICK
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and not lasso_active
            and event.button == 1
        ):  ##check for left mouse clicks
            onLeftClick(event.pos[0], event.pos[1])

        # # RIGHT CLICK
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and not lasso_active
            and event.button == 3
        ):  # Check for right mouse clicks
            onRightClick(event.pos[0], event.pos[1])

        if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            gravity_enabled = not gravity_enabled

    mouse_trajectory.append(
        (mouseX, mouseY)
    )  ##populating an array with the mouse's position this frame
    if (
        len(mouse_trajectory) > 20
    ):  ##if our mouse_trajectory array is logner than 20 store positions, pop the oldest position. We don't need it.
        mouse_trajectory.pop(0)

    ##UPDATE POSITIONS
    # Make a copy of the ball list
    for ball in balls.copy():
        # If a ball is inactive
        if not ball.active:
            # Remove it from the original list (this will stop it from being rendered)
            # The copy list is necessary because if you looped through the original list,
            #     I think it would break because the index you're on is no longer a thing
            balls.remove(ball)

    # Add gravity interaction to each ball
    for i, ball_1 in enumerate(balls):
        acceleration_x = 0
        acceleration_y = 0

        for j, ball_2 in enumerate(balls):
            # doesnt have gravity on itself
            if i == j:
                continue

            # gravity formula
            # if gravity_enabled:
            distance_x = ball_2.x_pos - ball_1.x_pos
            distance_y = ball_2.y_pos - ball_1.y_pos
            squared_distance = distance_x * distance_x + distance_y * distance_y
            force = gravity * ball_2.mass / (squared_distance + 0.001)

            # get acceleration on each direction
            acceleration_x += force * distance_x
            acceleration_y += force * distance_y

        # apply half of ball_2's force into ball_1 (to make the game not so easy, sorry!)
        if gravity_enabled:
            ball_1.apply_forces(acceleration_x / 2, acceleration_y / 2)

    # update each ball after calculations
    for ball in balls:
        old_x, old_y = ball.get_center()

        ball.update_position(mouseX, mouseY)
        ball.increase_burn()

        if ball.shot_frames > 0 and not ball.scored and old_y < HOOP_Y <= ball.y_pos:
            frac = (HOOP_Y - old_y) / (ball.y_pos - old_y)
            crossing = old_x + frac * (ball.x_pos - old_x)

            if HOOP_LEFT <= crossing <= HOOP_RIGHT:
                ball.scored = True
                ball.update_image()

        if not gravity_enabled:
            ball.apply_gravity_and_velocity(WINDOW_HEIGHT, WALL_THICKNESS)

    # keep track of the trails for each ball
    for ball in balls:
        ball.trail.append((ball.x_pos, ball.y_pos))
        # remove excess trail points
        if len(ball.trail) > MAX_TRAIL:
            ball.trail.pop(0)

    ashes[:] = [ash for ash in ashes if ash.active]

    for ash in ashes:
        ash.apply_gravity_and_velocity(WINDOW_HEIGHT, WALL_THICKNESS)
        ash.update_position()
        ash.increase_burn()

    handle_ball_collisions(
        balls, hitMarkerSound=hitmarkerSound
    )  ##this makes da balls bounce off eachother
    if any(ball.level >= 38 for ball in balls):
        won = True
    handle_ash_collisions(ashes)

    for ball in balls:
        ball.bounds(WINDOW_WIDTH, WINDOW_HEIGHT, WALL_THICKNESS)
    for ash in ashes:
        ash.bounds(WINDOW_WIDTH, WINDOW_HEIGHT, WALL_THICKNESS)

    # Random explosion
    # These are kinda weird / unexpected -> turn in to meteorites?
    if random.randint(1, 4000) < 30:
        x = random.randint(WALL_THICKNESS + 10, WINDOW_WIDTH - WALL_THICKNESS - 10)
        y = random.randint(WALL_THICKNESS + 10, WINDOW_HEIGHT - WALL_THICKNESS - 10)

        explosion = ExplosionEffect(x, y, random.randint(30, 50), explosionSound)
        explosions.append(explosion)

        explosion_knockback(x, y, balls, explosion.blast_radius)

    # Update active explosions
    for explosion in explosions:
        explosion.update()

    # Filter completed explosions
    explosions = [explosion for explosion in explosions if not explosion.finished]

    ##RENDER ALL STUFF
    bg_img = pygame.image.load("assets/background/background.png")
    screen.blit(bg_img, bg_img.get_rect())

    draw_max_balls()
    draw_attraction_bar()

    for ball in balls:
        # generate random colors for the trail
        random_r = random.randint(0, 255)
        random_g = random.randint(0, 255)
        random_b = random.randint(0, 255)
        trail_color = (random_r, random_g, random_b)
        # ignore first position change
        if len(ball.trail) >= 2:
            # render trail
            for i in range(1, len(ball.trail)):
                min_width = 2
                max_width = 6
                # keep track of the progress of the trail
                progress = i / (len(ball.trail) - 1)
                # calculate width of the current trail point
                width = max(1, int(min_width + progress * (max_width - min_width)))
                # draw trail on screen
                pygame.draw.line(
                    screen, trail_color, ball.trail[i - 1], ball.trail[i], width
                )
        # render ball
        ball.draw(screen)
    for ash in ashes:
        ash.draw(screen)

    screen.blit(
        help_text, (WALL_THICKNESS + 5, WALL_THICKNESS + 5)
    )  ##little instructions in the top left corner
    screen.blit(help_text2, (WALL_THICKNESS + 5, WALL_THICKNESS + 25))
    screen.blit(help_text3, (WALL_THICKNESS + 5, WALL_THICKNESS + 45))
    screen.blit(help_text4, (WALL_THICKNESS + 5, WALL_THICKNESS + 65))
    screen.blit(help_text5, (WALL_THICKNESS + 5, WALL_THICKNESS + 85))
    screen.blit(gravity_text, (WALL_THICKNESS + 5, WALL_THICKNESS + 155))
    screen.blit(
        ballsPerClickHelpText, (WALL_THICKNESS + 5, WALL_THICKNESS + WINDOW_HEIGHT - 50)
    )
    screen.blit(
        ballsPerClickPlusMinusHelpText,
        (WALL_THICKNESS + 600, WALL_THICKNESS + WINDOW_HEIGHT - 40),
    )

    if len(lasso_points) >= 2:
        alpha = 255

        if not lasso_active and lasso_fade_start is not None:
            elapsed = pygame.time.get_ticks() - lasso_fade_start

            alpha = max(0, 255 * (1 - elapsed / LASSO_FADE_TIME))

            if elapsed >= LASSO_FADE_TIME:
                lasso_points = []
                lasso_fade_start = None
        if lasso_points:
            lasso_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

            pygame.draw.lines(
                lasso_surface, (255, 0, 0, int(alpha)), False, lasso_points, 5
            )

            screen.blit(lasso_surface, (0, 0))

    # Render ball explosion
    for explosion in explosions:
        explosion.draw(screen)

    if won:
        win_text = big_font.render("You win!", True, "WHITE")
        screen.blit(win_text, win_text.get_rect(center=(WINDOW_WIDTH // 2, 100)))
    else:
        clock_text = big_font.render(countdown_text, True, "RED")
        screen.blit(clock_text, clock_text.get_rect(center=(WINDOW_WIDTH // 2, 100)))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
