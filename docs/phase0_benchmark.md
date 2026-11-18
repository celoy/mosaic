# Phase 0 benchmark (2026-11-18)

Goal: sweeps per second of random sequential updates with one edge reaction on an n × n hex grid,
in the Godot web export, to choose the grid size (target ≥ 4 sweeps/s at ×1).

## Set-up

- Godot 4.7.2 stable, GDScript, Compatibility renderer; web export, release template, thread support off.
- Machine: Apple M3 Pro, 36 GB; Google Chrome 157.0.8103.79, run headless (`--headless=new`).
- Grid: odd-r offset hexes, periodic (torus), 3 n² edges. 3 colours, uniform random start, seed 7.
- Kernel (`game/bench/bench_sim.gd`): pick an edge uniformly, look up its ordered colour pair in a
  table folding both orientations, apply. One sweep = 3 n² attempts.
- Scene (`game/bench/bench.gd`): per frame, simulate for a 12 ms budget, upload colours as an
  R8 texture (one call), draw all hexes with one MultiMesh coloured in the shader. 1 s warm-up,
  then 6 s measured per size.
- Native baseline (`game/tests/bench_cli.gd`): headless editor binary, simulation only.

## Results

| n | edges | web+render sweeps/s (fps) | native sim sweeps/s, invade | native sim sweeps/s, exchange |
|---|---|---|---|---|
| 40 | 4 800 | 2244 (60) | 2622 | 2209 |
| 48 | 6 912 | 1557 (60) | 1785 | 1619 |
| 64 | 12 288 | 892 (60) | 1008 | 910 |
| 96 | 27 648 | 388 (60) | 454 | 406 |
| 128 | 49 152 | 219 (60) | 254 | 228 |

Cost per attempt: ≈ 0.066 µs in Chrome (release wasm), ≈ 0.080 µs native editor build
(invade, mostly lookup misses), ≈ 0.089 µs native with the exchange reaction A|B → B|A
(fires on ≈ 21 % of attempts, close to the mean-field 2 x₀x₁ = 2/9).

Reactions: "invade" is A|B → A|A (the scene and the web numbers); colour 1 is invaded within the
warm-up, so the measured attempts are mostly misses. "exchange" keeps firing forever and bounds
the hit-path cost (+10 %). The sim used its full 12 ms budget (72 % of wall time) and frames held 60 fps, so texture upload and drawing fit in the remaining ~4.7 ms.

## Decision proposed

Throughput at 64 × 64 is ≈ 900 sweeps/s in Chrome, > 200 × the 4 sweeps/s target. Keep
**64 × 64** (performance does not bind; a larger grid is affordable but cells become small on a
laptop screen). The real engine (several reactions, rate acceptance, vertex rotations, lamp
counters, 2 n² extra vertex elements) will cost more per attempt; a 10–20 × slowdown still leaves
a wide margin.

Caveat: headless Chrome uses software WebGL and the user's real Chrome was not measured here;
the simulation numbers should transfer, the frame rate on a real GPU can only be better.
