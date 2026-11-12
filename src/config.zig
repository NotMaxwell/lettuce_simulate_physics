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
