# Configs

Each file is a ready-to-run scenario: `build/lettuce configs/disk.ini`.

| File | Scenario | Default n | What it shows |
|---|---|---|---|
| plummer.ini | plummer | 10000 | virialised star cluster in Henon units |
| kepler.ini | kepler | 2 | two-body orbit, integrator accuracy check |
| cold_collapse.ini | uniform | 20000 | cold sphere collapse and virialisation |
| disk.ini | disk | 20000 | rotating exponential disk with central mass |
| collision.ini | collision | 40000 | two inclined disk galaxies colliding |
| solar.ini | solar | 2009 | real Solar System plus asteroid belt (AU, yr, Msun) |
| systems.ini | systems | 10000 | Plummer cluster of stars with planetary systems |

On a 4-core CPU the defaults take 1 s (kepler) to 2 min (cold_collapse); collision.ini (40k
bodies, direct) is sized for a GPU and takes ~12 min on that CPU, so add `n=4000` there.

## Format

`key = value` lines; `#` or `;` start a comment (whole line or trailing). `[section]` prefixes
the following keys with `section.` (so `[disk]` then `scale = 2` is `disk.scale`); `[]` or
`[general]` returns to top-level keys. Top-level keys are the run settings (`n`, `dt`, `steps`,
`method`, ...); any key containing a dot is a scenario parameter. Unknown top-level keys are an
error. Integers may be written `1e6`, `1_000_000` or `1000000`; booleans accept true/false/yes/no/on/off/1/0.

## Overrides

Arguments after the file are `key=value` and win over the file, using the same key names:

    lettuce configs/disk.ini n=1000000 method=tree disk.thickness=0.1

A scenario parameter the chosen scenario does not read (e.g. a typo like `plummer.cutt`) stops
the run with an error. The effective configuration is written to `<out_dir>/config.ini`, which
reruns the same thing.
