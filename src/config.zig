pub const base_w: f32 = 1029;
pub const base_h: f32 = 720;
pub const wall: f32 = 10;
pub const tick_hz: f32 = 60;

pub const max_level: u8 = 38;
pub const start_level: u8 = 4;
pub const max_trail = 30;
pub const max_radius: f32 = 100;
pub const spawn_radius_min: f32 = 25;
pub const spawn_radius_max: f32 = 45;

pub const mutual_g: f32 = 0.6;
/// Ball count up to which mutual gravity is at full (original) strength.
pub const gravity_full_count: f32 = 60;
pub const bounce_stop: f32 = 0.3;

pub const room_temp: f32 = 21;
pub const ignition_temp: f32 = 100;
pub const flame_temp: f32 = 400;
pub const max_burn: f32 = 100;
pub const base_conduct_mult: f32 = 0.0005;
pub const impact_heat_mult: f32 = 1.5;
pub const base_explosion_heat: f32 = 350;
pub const base_explosion_force: f32 = 10_000;

pub const hoop_left: f32 = 335;
pub const hoop_right: f32 = 375;
pub const hoop_y: f32 = 450;

pub const countdown_seconds = 120;
pub const attraction_energy_max: f32 = 75;
pub const lasso_fade_ms: f32 = 1000;

/// Fraction of the play area the balls may cover before the area grows.
pub const max_fill: f32 = 0.30;

// Simulation tuning (physics.zig / effects.zig). Tick counts assume 1 tick = 1 Python frame.
pub const attraction_strength: f32 = 0.1;
/// Python: random.randint(1, 4000) < 30.
pub const random_explosion_chance: f32 = 29.0 / 4000.0;
pub const random_explosion_size_min = 30;
pub const random_explosion_size_max = 50;
pub const explosion_size_mult: f32 = 10;
pub const blast_radius_mult: f32 = 0.8;
pub const explosion_frames = 12;
pub const explosion_frame_ticks = 6;
pub const explosion_ticks: u16 = explosion_frames * explosion_frame_ticks;

pub const fire_particle_life: u16 = 20;
pub const fire_particle_vy: f32 = -3;
pub const max_particles = 40_000;

pub const ash_radius: f32 = 2;
pub const ash_lifetime_ticks: u16 = 180;
/// Grid step between ashes spawned inside a burnt-out ball.
pub const ash_spawn_step = 8;
pub const max_ashes = 20_000;

/// Per-tick easing of the play area towards its target size.
pub const area_grow_rate: f32 = 0.15;
pub const area_shrink_rate: f32 = 0.01;
