# physics_simulation

A little pygame merge-sandbox with bouncy balls. Click anywhere empty to spawn a ball with random size, color, mass, bounciness, and friction. Balls will merge with balls of the same image to create bigger balls, and a higher level. 38 is the highest level. Balls bounce off the walls and off each other (mass-based elastic collisions, so heavy balls shove light ones around). Balls are gravitationally attracted to each other by the mass. The bigger the ball, the bigger the pull. Balls will turn into ashes after burning for long enough
Balls can also be set on fire, which can transfer to other balls based on their temperature (°C) and kinetic energy. Additionally balls can explode, which also transfers heat. (Both are based on mass & distance) To put out the fires, click rapidly on the ball before it burns up.

## Controls

- **Left click empty space:** spawn a random ball
- **Right click a ball:** set the ball on fire
- **Left click a ball:** cool it down; hold and drag to move it, then release to throw it
- **Hold Shift and drag around a ball:** lasso it to cause an explosion
- **Hold G and left click:** attract balls toward the mouse while energy lasts
- **Press F:** switch between downward gravity and attraction between balls
- **Press M:** toggle background music
- **Press + or -:** increase or decrease balls spawned per click

## Zig rewrite (branch `zig-rewrite`)

The game is being rewritten in **Zig 0.17 + SDL3** to run 1000+ balls at 60 fps. The Python
files remain as the behavioural reference.

```bash
zig build run                 # play (optimized build)
zig build bench -- 1000 600   # headless physics timing: <balls> <ticks>
zig build test-bench          # physics unit tests
```

Needs Zig 0.17 and the SDL3 development package. Extra keys in the Zig version:
**T** stress test (+1000 balls, merging off), **N** toggle merging, **P** perf overlay,
**Ctrl with + / -** change balls per click by 25.

## Setup (original Python version)

**Python 3.13** with **pygame 2.6.1**.

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
## Layout

- `main.py` — pygame setup, event handling, and the main game loop
- `ball.py` — the `Ball` class: gravity, wall bounces, floor friction, mouse clicking for putting out fires
- `ash.py` — the `Ash` class: gravity, wall collisions, and fades away
- `utils.py` — ball-to-ball and ash-to-ash collision handling
- `particle.py` — reusable particle class for an effect controller to use
- `fire_effect.py` — fire effect that is used when a ball is on fire
- `explosion_effect.py` — explosion effect that is used when you lasso around a ball
- `constants.py` — constants that are used around the project
