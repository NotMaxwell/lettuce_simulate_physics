# physics_simulation

A merge-sandbox with bouncy balls, written in **Zig 0.17 + SDL3** and built to run thousands of
balls at 60 fps. Click anywhere empty to spawn a ball with random size, mass, bounciness and
friction. Balls merge with balls of the same image into a bigger, higher-level ball; 38 is the
highest level. Balls bounce off the walls and each other (mass-based elastic collisions) and are
gravitationally attracted to each other. Balls can be set on fire, which spreads by temperature
and impact; explosions knock balls away and set the ones near the centre alight. Burning balls
turn to ash unless you click them to cool them down. The play area grows as you add balls.

It started as a Python/pygame learning project; that version lives on in [`python/`](python/).

## Run

A prebuilt Linux x86-64 binary is in `bin/`. Run it from the repo root so it finds `assets/`:

```bash
./bin/lettuce-linux-x86_64
```

It needs SDL3 installed and a CPU with AVX2 (2013 or newer), and was built on Arch against a
recent glibc, so older distros may need to build from source.

## Build

Needs Zig 0.17 and the SDL3 development package.

```bash
zig build run                 # play (optimized build)
zig build run -Doptimize=debug  # with bounds and overflow checks
zig build bench -- 1000 600   # headless physics timing: <balls> <ticks>
zig build test-bench          # physics unit tests
```

## Controls

- **Left click empty space:** spawn balls (balls per click shown bottom-left)
- **Right click a ball:** set it on fire
- **Left click a ball:** cool it down; hold and drag to move it, release to throw it
- **Hold Shift and drag around balls:** lasso them to make them explode
- **Hold G and left click:** attract balls toward the mouse while energy lasts
- **F:** switch between downward gravity and attraction between balls
- **+ / -:** change balls per click (hold Ctrl to step by 25)
- **T:** stress test, adds 1000 balls and turns merging off
- **N:** toggle merging
- **P:** toggle the performance overlay
- **M:** toggle background music

## Layout

- `src/main.zig` — window, input, fixed-timestep game loop
- `src/physics.zig` — simulation tick: SIMD gravity, grid collisions, merging, heat, hoop
- `src/effects.zig` — fire particles, ash, explosions
- `src/render.zig` — batched GPU drawing, zooming camera, HUD
- `src/audio.zig` — sound effects and music
- `src/world.zig`, `src/config.zig` — shared data and tuning constants
- `vendor/` — single-header image and MP3 decoders
- `python/` — the original pygame version

## Original Python version

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r python/requirements.txt
python python/main.py         # run from the repo root so it finds assets/
```
