# Phase 1 notes (2026-11-18): Python twin, science checks, viewer

Raw outputs (gitignored): `runs/science/*_20261118-1129*`, `*_20261118-1131*`; run log `runs/LOG.md`.
Reproduce: `python tools/science.py all` (≈ 4 min on an M3 Pro), `pytest`.

## Model conventions that go beyond the brief

1. **Channels.** Every (reaction, orientation) matching an element fires at rate k. So a
   symmetric input (A|A) fires at 2k, and an achiral vertex reaction is the reaction plus its
   mirror, each at rate k. The mean-field equations use the same counting
   (orientations: 1 cell, 2 edge, 3 chiral vertex, 6 achiral vertex).
2. **Acceptance.** An attempt fires with probability (sum of matching channel rates) / K, with
   K = the largest total rate of any element state, and picks a channel in proportion to its
   rate. One sweep advances time by 1/K. The brief's rule ("choose ∝ rate, then apply with
   rate/max_rate") gives a firing probability ∝ k²/Σk when several reactions match, which is
   not a Markov chain with rates k. The two agree when each state matches one channel.
3. **Shuffled control.** Shuffling once per sweep (the brief) is too coarse: one sweep is 1/K
   time units, and cells change O(1) times within it, so correlations build up between shuffles.
   On the test set this gives errors of 0.05 against mean field. The default is now
   per-attempt mixing (the element's cells are drawn at random), which matches mean field to
   < 0.015 on 64 × 64 and < 0.02 on 128 × 128 (test 4). Per-sweep shuffling is kept as an option.
4. **Vertices and chirality.** A vertex's cells are listed clockwise as seen on screen
   (y down). This is checked geometrically in the tests.
5. **Goals.** Durations are in sim time units, not wall seconds. Each goal latches once met,
   and the level is won when all goals have latched. Failure conditions use the same types.
6. **Grid.** The grid is periodic (torus), with the same odd-r layout as the Godot benchmark.

Speed: 0.12–0.15 µs per attempt in pure Python, ≈ 3–4 ms per sweep at 64 × 64. numba is not needed yet.

## Science checks

**1. Voter (n = 128, 3 seeds, t ≤ 500).** The unlike-edge density ρ falls from 0.5 to 0.30
(t = 1), 0.24 (5), 0.20 (20), and then stays at 0.17–0.19 from t = 50 to 324, with run-to-run
fluctuations of ±0.015. An independent minimal link-update implementation gives the same numbers.
In this range neither 1/ρ ∝ ln t (R² 0.78) nor a power law (exponent −0.09, R² 0.80) fits
well. The expected 1/ln t decay is too slow to see under interface noise. For the game: voter
coarsening is visibly slow and noisy; domains grow but the total interface barely changes after t ≈ 50.

**2. Cyclic invasion + exchange ε (4 seeds).**

| n | t_max | ε = 0 | 0.1 | 0.3 | 1 | 3 | 10 | 30 | well-mixed |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 1000 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 4/4 (t≈140) | 4/4 (t≈60) | 4/4 (t≈87) |
| 64 | 300 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 (t≈116) | 2/4 (t≈286) |

(number of runs that lost a species). The result is qualitatively that of Reichenbach,
Mobilia and Frey: mobility destroys coexistence above a threshold. The threshold in ε grows
with N (between 3 and 10 at N = 1024, about 30 at N = 4096), consistent with a critical
mobility M ∝ ε/N. The patterns differ from theirs, though. Our model is the zero-sum
version, with no empty sites, and gives irregular domains, not spirals. **Spirals as promised
in the pitch need May–Leonard-type reactions (an "empty" colour with reproduction into it),
which use 3 colours plus empty = 4 colours.**

**3. Chiral rotation (0,1,2) → (2,0,1).**
- Alone, chiral or achiral, it produces no pattern: ρ = 0.67 and 23 % trichromatic vertices,
  the values of a uniform random state. Each firing is a bijection between configurations, so
  the uniform state is stationary.
- Added to cyclic invasion (k = 0.3), it slightly raises the interface density (0.30 → 0.32).
  Chiral and achiral are indistinguishable (0.322 vs 0.324).
- My chirality order parameter (#cw − #ccw trichromatic vertices) is **identically zero** on a
  torus, by a Sperner/degree argument. Net handedness cannot build up.

The consequence for design: with these reactions, chirality is invisible to the player. A
chiral level needs a reaction whose outcome is not itself a rotation (for example one
that converts colours), so the two handednesses do different things. Not explored yet.

**4. Levels, normal vs well-mixed, no switches (4 seeds, t ≤ 200).**
- T01 voter: well-mixed is neutral drift (mean field dx/dt = 0); the grid is slow coarsening.
  One run of four reached extinction in each mode.
- T02 RPS + exchange: the grid coexists in 4/4 runs; well-mixed lost a colour in 1/4 (t ≈ 99).
- T03 RPS + chiral rotation: the grid coexists in 4/4 runs; well-mixed lost a colour during
  burn-in in 3/4.
- Comparing with mean field at one time is meaningless for these sets, because zero-sum
  cyclic invasion has neutral cycles. The useful comparison is extinction statistics:
  coexistence on the grid against extinction when well-mixed is a real spatial effect.

## Test levels

T01 voter, T02 RPS + exchange, T03 chiral (the brief's L04). None has been checked for
solvability (that is Phase 2). One observation from the viewer smoke test: an arbitrary single swap in
T02 (exchange 0|1 → 1|0 becomes 0|1 → 1|1 at rate 2) wipes out two colours by t ≈ 3. T02
falls to one switch.

## Addendum (same day): empty colour, May–Leonard

Colour 3 is used as empty space. Reactions: selection i|i+1 → i|E (σ = 1), reproduction
i|E → i|i (μ = 1), exchange at rate ε, either between all unlike pairs ("all") or only hops
into empty sites ("holes"). Run: `runs/science/mayleonard_20261118-121110`.

| exchange | ε = 0 | 0.3 | 1 | 3 | 10 |
|---|---|---|---|---|---|
| all, extinct (2 seeds, t ≤ 200, n = 64) | 0/2 | 0/2 | 0/2 | 1/2 | 2/2 (t ≈ 11) |
| holes | | 0/2 | 0/2 | 0/2 | 0/2 |

- The empty fraction stays low on the grid (5–9 %), against 25 % at the mean-field fixed point.
- At 64 × 64 there are domains but no recognisable spirals, for every ε tried.
- At 128 × 128 with ε = 1 (all), curved rotating arms appear: rough spirals. The spiral
  wavelength (∝ √ε) is comparable to a 64-cell grid, so spirals need about 128 × 128, or
  reaction rates that shorten the wavelength.
- The Phase 0 benchmark gave 219 sweeps/s at 128 × 128 in Chrome, so the larger grid is
  affordable in Godot.

T04 (levels/T04_mayleonard.json) is the "holes" variant with ε = 1 (9 reactions, so the
panel fits). Try it with `--n 128` to compare.

## Addendum: active-set (incremental) algorithm vs rejection

Question (Chris): would it be faster or leaner to keep tables of which elements can react,
updated after each firing, instead of drawing among all elements and rejecting?

Active-set algorithm (`model/mosaic_model/sim_active.py`, `game/tests/bench_active.gd`):
- Keep the list of elements whose colours match at least one channel, and draw attempts
  only from it; time advances by 1/(K·M) per attempt, M = list size.
- After a firing, re-check every element that touches a changed cell: 13 per cell
  (1 cell, 6 edges, 6 vertices).
- Same Markov chain as the rejection algorithm.

**Equivalence** (`tools/bench_active.py`: 24 seeds per algorithm, n = 32, 4 reaction sets
covering all shapes and a chiral count-changing vertex reaction). The 15 observables agree
within statistical error; the largest |z| is 1.87.

**Speed** (wall time per unit of sim time, both algorithms from the same state).

| case | n | fraction of attempts that fire (rejection) | Python speed-up | GDScript speed-up |
|---|---|---|---|---|
| T01–T04 stationary starts | 64 | 15–20 % | 0.31–0.49 | 0.35–0.36 (T02/T03 edges) |
| T01–T04 stationary starts | 128 | 15–21 % | 0.34–0.66 | 0.35–0.36 |
| single straight front | 64 | 3.5 % | 2.0 | 2.2 |
| single straight front | 128 | 1.9 % | 4.0 | 4.0 |

(speed-up = rejection time / active time; < 1 means the active set is slower.)

Conclusion: in the states the game lives in, 15–30 % of attempts fire, and a firing costs
about one attempt plus several re-checks. Here the active set is about 3× slower, in both
Python and GDScript. It wins only when activity is sparse (break-even near 5–10 % of
elements active), and there the absolute cost of rejection is already small
(GDScript: 1.4 ms per time unit at 64², 5.4 ms at 128²). Memory is negligible either way
(≈ 0.5 MB of index arrays at 128²).

Decision proposed: keep the rejection algorithm. If late, coarsened states ever matter for
speed (solver runs to extinction), a hybrid could count active elements now and then and
switch below ~5 %.

Absolute GDScript cost (native headless, rejection): RPS + exchange (T02) 4.6 ms per
time unit at 64², 18 ms at 128². At ×16 (16 time units per second) that is 7 % / 29 % of
a CPU core. Both fit.

## Addendum: world shape (run `runs/science/shapes_20261118-123517`)

Four worlds of about 4000 cells: the rectangular torus 64 × 64 (glued), the same block with
open sides, a hexagon of side 37 with opposite sides glued (a hexagonal torus), and the same
hexagon open. Means over 4 seeds.

| | torus 64 | rect 64 open | hex torus 37 | hexagon 37 open |
|---|---|---|---|---|
| cells with < 6 neighbours | 0 % | 6.2 % | 0 % | 5.4 % |
| voter ρ at t = 5 / 20 / 50 | .232/.210/.202 | .242/.184/.150 | .242/.195/.174 | .237/.189/.159 |
| RPS ε = 0, 3, 10: lost a species (t ≤ 300) | 0/4 each | 0/4 each | 0/4 each | 0/4 each |
| May–Leonard (all, ε = 1): lost a species | 0/4 | 0/4 | 0/4 | 1/4 |
| May–Leonard x3 at end (all / holes) | .089/.062 | .082/.052 | .087/.053 | .062/.055 |
| chiral count-changing set, x1 at t = 60 | .983 | .986 | .990 | .989 |

- **Shape matters mostly for slow coarsening.** Voter interfaces fall faster with open
  sides (ρ = 0.15–0.16 vs 0.20 at t = 50). The rectangular torus is the slowest; the hexagonal
  torus sits in between.
  Plausible reasons (not tested): domains can wrap around a torus, forming straight bands
  that never shrink, and the rectangular torus has two short directions while the hexagonal
  one has three equal ones. The difference is about 2σ with 4 seeds.
- **Fast reactions don't care.** For RPS, May–Leonard and the chiral set, every world gives
  the same outcome within noise.
- **Net chirality stays zero in every world.** With open sides, #cw − #ccw trichromatic
  vertices equals the winding of the colours along the border, which was zero in these runs.
- The 1:3:2 weights of the mean-field equations are slightly off with open sides
  (border cells have fewer edges and vertices); 5–6 % of cells at this size.
