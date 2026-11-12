import math

import pygame

from constants import BASE_CONDUCT_MULT, IMPACT_HEAT_MULT


def calc_mouse_vector(mouse_trajectory):
    x_speed = 0
    y_speed = 0
    if len(mouse_trajectory) > 19:
        x_speed = (mouse_trajectory[-1][0] - mouse_trajectory[0][0]) / len(
            mouse_trajectory
        )
        y_speed = (mouse_trajectory[-1][1] - mouse_trajectory[0][1]) / len(
            mouse_trajectory
        )
    return x_speed, y_speed


def handle_ball_collisions(balls, hitMarkerSound=None):
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):  ##check every pair of balls exactly once
            ball1 = balls[i]
            ball2 = balls[j]

            x_dist = ball2.x_pos - ball1.x_pos
            y_dist = ball2.y_pos - ball1.y_pos
            distance = math.sqrt(x_dist**2 + y_dist**2)
            if 0 < distance < ball1.radius + ball2.radius:  ##the balls are overlapping
                if ball1.level == ball2.level and ball1.level < 38:
                    ##MERGE!
                    if hitMarkerSound is not None:
                        hitMarkerSound.play()
                    if ball1.radius > ball2.radius:
                        ball2.active = False
                        ball1.level += 1
                        ball1.level = min(ball1.level, 38)
                        ball1.temperature += ball2.temperature
                        ball1.temperature *= 0.75
                        ball1.radius = min(100, ball1.radius + math.log1p(ball2.radius))
                        ball1.update_image()
                    else:
                        ball1.active = False
                        ball2.level += 1
                        ball2.level = min(ball2.level, 38)
                        ball2.temperature += ball1.temperature
                        ball2.temperature *= 0.75
                        ball2.radius = min(100, ball2.radius + math.log1p(ball1.radius))
                        ball2.update_image()
                    continue

                normal_x = x_dist / distance  ##direction pointing from ball1 to ball2
                normal_y = y_dist / distance
                overlap = (ball1.radius + ball2.radius) - distance

                ##push the balls apart so they aren't inside eachother. a held ball shouldn't get shoved off the mouse tho
                ball1.x_pos -= normal_x * overlap / 2
                ball1.y_pos -= normal_y * overlap / 2
                ball2.x_pos += normal_x * overlap / 2
                ball2.y_pos += normal_y * overlap / 2

                ##elastic collision, heavier balls shove lighter balls around more
                rel_speed = (ball1.x_speed - ball2.x_speed) * normal_x + (
                    ball1.y_speed - ball2.y_speed
                ) * normal_y
                tan_speed = (ball1.x_speed - ball2.x_speed) * -normal_y + (
                    ball1.y_speed - ball2.y_speed
                ) * normal_x

                ball1.angular_speed += tan_speed / ball1.radius * 0.5
                ball2.angular_speed -= tan_speed / ball2.radius * 0.5

                if (
                    rel_speed > 0
                ):  ##only bounce if the balls are actually moving towards eachother
                    impulse = 2 * rel_speed / (ball1.mass + ball2.mass)
                    ball1.x_speed -= impulse * ball2.mass * normal_x
                    ball1.y_speed -= impulse * ball2.mass * normal_y

                    ball2.x_speed += impulse * ball1.mass * normal_x
                    ball2.y_speed += impulse * ball1.mass * normal_y

                # Check if impacted ball is on fire
                #     if so, they will set each other on fire
                #    This currently will immediately engulf all balls in flames if they are near each other
                # If you feel like making it spread cooler then please go ahead
                # (I'm burning out) - sure I did that with temperature :)

                temp_diff = ball1.temperature - ball2.temperature

                avg_density = (ball1.density + ball2.density) / 2
                avg_radius = (ball1.radius + ball2.radius) / 2
                conductivity = (
                    avg_density * avg_radius * BASE_CONDUCT_MULT
                )  # Value how fast heat gets transfered

                impact_heat = (
                    max(0.0, rel_speed) * IMPACT_HEAT_MULT
                )  # Kinetic impacts should generate heat

                # Transfer the heat based on the temp, conductivity, impact heat and ball mass
                if temp_diff > 0:  # Ball 1 is hotter
                    ball2.temperature += (
                        (temp_diff * conductivity + impact_heat) / (ball2.mass) * 10
                    )

                if temp_diff < 0:  # Ball 2 is hotter
                    ball1.temperature += (
                        (abs(temp_diff) * conductivity + impact_heat)
                        / (ball1.mass)
                        * 10
                    )

                # adding no temp difference, impact friction heat increase
                if temp_diff == 0:
                    ball1.temperature += impact_heat / (ball1.mass) * 10
                    ball2.temperature += impact_heat / (ball2.mass) * 10


def handle_ash_collisions(ashes, cell_size=1):
    # do a grid-based collision, more like sand than balls
    grid = {}

    for ash in ashes:
        cell_x = int(ash.x_pos // cell_size)
        cell_y = int(ash.y_pos // cell_size)
        grid.setdefault((cell_x, cell_y), []).append(ash)  # put them in grid cells

    for ash in ashes:
        cell_x = int(ash.x_pos // cell_size)
        cell_y = int(ash.y_pos // cell_size)
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):  # go through grid cells and do collisions
                nearby = grid.get((cell_x + offset_x, cell_y + offset_y), [])
                for other in nearby:
                    if other is ash:
                        continue
                    dx = other.x_pos - ash.x_pos
                    dy = other.y_pos - ash.y_pos
                    d2 = dx * dx + dy * dy
                    min_d = ash.radius + other.radius

                    if d2 > 0 and d2 < min_d * min_d:
                        d = math.sqrt(d2)
                        norm_x = dx / d
                        norm_y = dy / d
                        overlap = min_d - d
                        ash.x_pos -= norm_x * overlap / 2
                        ash.y_pos -= norm_y * overlap / 2
                        other.x_pos += norm_x * overlap / 2
                        other.y_pos += norm_y * overlap / 2
                        vel_x = other.x_speed - ash.x_speed
                        vel_y = other.y_speed - ash.y_speed
                        norm_v = vel_x * norm_x + vel_y * norm_y
                        if norm_v > 0:
                            continue

                        pow = -norm_v * 0.5
                        pow_x = pow * norm_x
                        pow_y = pow * norm_y
                        ash.x_speed -= pow_x
                        ash.y_speed -= pow_y
                        other.x_speed += pow_x
                        other.y_speed += pow_y


def point_in_polygon(point, polygon):
    x, y = point

    inside = False

    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if (yi > y) != (yj > y):
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside

        j = i

    return inside


# Useful if you want to add an animation from a single file of multiple sprites.
# Loads individual sprites from images with slicing
def load_spritesheet(filepath, rows, cols):
    sheet = pygame.image.load(filepath).convert_alpha()

    # Get the sheet & sprite sizes
    sheet_width, sheet_height = sheet.get_size()
    frame_width = sheet_width / cols
    frame_height = sheet_height / rows

    frames = []

    for row in range(rows):
        for col in range(cols):
            # Get the rectangles of each sprite from the sheet
            rectangle = pygame.Rect(
                frame_width * col, frame_height * row, frame_width, frame_height
            )
            frame = sheet.subsurface(rectangle)

            frames.append(frame)

    return frames
