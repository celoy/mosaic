# Mosaic Shift: Game Off 2026 project brief

Working title "Mosaic Shift" (not cleared, see Open questions). This file is the single brief for the whole jam, from the Python twin to the submitted Godot game. Claude Code reads it at the start of every session. It replaces all earlier briefs (sand frame, boat race).

## How to work with this file

- The user, Chris, is a fluid-mechanics professor and a solo part-time developer. He judges the dynamics by watching and playing. Make the physics visible and every parameter easy to change.
- Work only on the phase marked CURRENT in Status. Do not start the next phase without the user's go.
- At the end of each task: run the tests, then report what passes, every assumption you made that is not in this file, and anything here you think is wrong.
- Never change an entry in Decisions on your own; propose the change and wait.
- At the end of each session, append one dated line to Status.
- Do not add features that are not in the current phase. When in doubt, ask.

## Status

- CURRENT: Phase 1
- 2026-11-18: brief written; nothing built yet.
- 2026-11-18: Phase 0 done: repo celoy/mosaic (MIT), .venv + pytest, Godot 4.7.2 web benchmark ≈ 890 sweeps/s at 64 × 64 in headless Chrome (docs/phase0_benchmark.md); grid size 64 × 64 proposed. Waiting for go on Phase 1.
- 2026-11-18: Phase 1 built (grid 64 × 64 confirmed by Chris): mosaic_model, tests 1–7 pass, science checks 1–4 run (docs/phase1_notes.md), pygame viewer, test levels T01–T03. Gate B pending: Chris plays the viewer.
- 2026-11-18: Chris accepted the rate rule and per-attempt shuffled control. Added empty colour (T04 May–Leonard), viewer at 1 time unit/s with partial sweeps, Pause/speed/Restart buttons, swaps only while running, arrow thickness ∝ k, --n grid override. Spirals visible at n = 128, not at 64.
- 2026-11-18: Chris: colours only, no special empty colour; no limit on the number of colours for now; no hex borders. Active-set vs rejection benchmark (Python + GDScript): rejection kept (active set ~3× slower at 15–30 % firing, faster only below ~5–10 % active).
- 2026-11-18: World shapes added (hex torus, hexagonal torus, open versions; grid spec "type": "hexagon", "side", "periodic"). Shape changes voter coarsening speed, not RPS/May–Leonard/chiral outcomes. Viewer: Chris's palette + palette menu (okabe_ito, tol_bright, greyscale), symbols off by default, hex equation blocks, periodic worlds tiled to the screen edge, thin chevron arrows.

## Deadline and jam

- GitHub Game Off 2026, theme SHIFT. Submissions close 1 December 2026 at 21:37 UTC (13:37 PST). Target submission: Monday 30 November, evening (Paris time).
- The source must be in a public GitHub repository. Any engine is allowed.
- Rated categories: overall, gameplay, graphics, audio, innovation, theme interpretation.
- Before submitting, the user checks the jam page for any rule on AI-assisted code.

## The game in one page

Pitch. A living mosaic of coloured cells is held in balance by a few reactions. You cannot paint the cells. You can only shift colours between the reactions, live, and watch the mosaic tip toward a goal: wipe out a colour, hold a balance, reach a mixture.

The world. A large grid of cells (hexagonal first, foam later if time), with edges between neighbouring cells and vertices where three cells meet. Each cell has a colour from a palette of 3–4.

Reactions (given by the level: number, shape and rate are fixed):
- Cell reaction: A → B.
- Edge reaction: A|B → C|D, on any pair of neighbouring cells, either orientation.
- Vertex reaction: (A, B, C) → (D, E, F) around a vertex, read clockwise. Chirality matters: the mirror arrangement does not match unless the level says the reaction is achiral.

The only verb. Every colour in every reaction equation is a block. The player swaps any two unlocked blocks, anywhere among the equations, at any time, while the mosaic runs. The multiset of blocks is conserved, so every move is a permutation: a shift of colours between reactions. This is the theme.

The dynamics. Random sequential updating: pick a cell, edge or vertex at random, look up the reactions matching its colours, apply one according to the rates. Large grids make the large-scale behaviour nearly deterministic; the texture stays alive.

Start. Each level starts from a stationary state of its initial arrangement: fronts wandering, spirals turning, population fractions steady. The first switch perturbs a living system.

Goals only, no score. Goal types: extinct (a colour disappears), hold (all fractions stay within bands for a duration), reach (fractions within a tolerance of a target, held for a few seconds). Some levels add a failure condition (for example, a colour going extinct during a hold).

Legibility. A lamp beside each equation glows with that reaction's recent firing rate. Optional overlay: a brief flash on the grid where reactions fire. A population chart shows the fractions over time with the goal bands drawn on it. The theme is named in the tutorial ("shift a colour between reactions").

Why it is hard: effects are delayed and spread over the whole grid; spatial dynamics defeat well-mixed intuition (rock-paper-scissors spirals, voter-model coarsening); the blocks are conserved, so fixing one reaction changes another; timing matters, because the same switch made at a different moment meets a different landscape.

Fun hypothesis (falsifiable in Phases 1–2):
- In most levels a solution needs at least two switches with a wait between them, so timing matters.
- In at least a third of levels the well-mixed (mean-field) prediction is wrong, so the grid is not decoration.
- A correct solution fails in fewer than 10 % of random seeds on the full grid.
- A dithering bot (switching back and forth quickly) does not solve most levels trivially. If it does, add a switch cooldown.

Kill criterion, at Gate B (Sunday 22 November): if most test levels fall to a single switch, or the mean-field equations predict every outcome, the spatial game is decoration. Redesign the reaction sets before building levels.

Out of scope for the jam: painting cells, editing reaction shapes or rates, scores and leaderboards, level editor, topological changes of the foam, touch controls, voice, 3D.

## Model

Graph. Cells, edges and vertices with full adjacency: cell → neighbouring cells, edge → its two cells, vertex → its three cells in clockwise order. Hexagonal grid generated in code; foam grids loaded from JSON (generated offline in Python).

Element counts (derived from Euler's formula for trivalent tilings, boundaries aside): per cell, about 3 edges and 2 vertices.

Update rule (one attempt):
1. Pick an element uniformly among all cells, edges and vertices that can host at least one current reaction shape (skip element types with no reaction).
2. Read its colours. Use a lookup table indexed by the colour tuple to find matching reactions: for edges, both orientations; for vertices, the three cyclic rotations of the clockwise triple (and the mirror only for achiral reactions).
3. Every (reaction, orientation) that matches is a channel firing at rate k (so A|A → … fires at 2k; an achiral vertex reaction is the reaction plus its mirror). Fire with probability (sum of matching channel rates) / K, where K is the largest total rate of any element state, and pick the channel in proportion to its rate (accepted 2026-11-18).
4. Rebuild the lookup tables whenever the player switches blocks (cheap).

Time: one sweep = one attempt per eligible element; the simulation clock advances by 1 / K per sweep (1 / (K · n_elig) per attempt, so partial sweeps are allowed). Speed control sets sim time units per wall second (viewer ×1 = 1 time unit per second).

Mean-field equations (derived, product measure: each cell independently has colour c with probability x_c): each reaction contributes to dx/dt its rate times the number of orientations (1 cell, 2 edge, 3 chiral vertex, 6 achiral vertex) times the product of the input fractions times the net change in colour counts, with attempt weights 1 : 3 : 2 for cell, edge and vertex reactions. Generate these equations automatically from any reaction set; they are used by the solver and by tests.

Shuffled control: a well-mixed mode in which every attempt draws the element's cells at random (equivalent to a permutation before every attempt; accepted 2026-11-18, because shuffling once per sweep leaves errors of about 5 % against mean field). It must agree with the mean-field equations on a large grid; the difference between normal and shuffled runs is the spatial effect.

Assumptions and regime: discrete-time random sequential updates approximate a continuous-time Markov chain when each sweep changes a small fraction of elements; mean-field is exact only for the shuffled control; extinction on a finite grid is always eventually possible, so "hold" goals have finite durations.

## Python twin (Phases 1 and 2)

Package `mosaic_model` (numpy; ask before adding numba if the loops are too slow). Holds the graph, reactions, lookup tables, update, shuffled control, mean-field generator, goals, and a JSON level loader shared with Godot.

Minimal viewer (pygame-ce, Phase 1): coloured hexes, reaction equations with lamps, click two blocks to swap them, pause and speed keys, population chart with goal bands. Enough for Chris to feel the game at Gate B.

Science checks (headless, CSV outputs):
1. Voter model (A|B → A|A and B|A → B|B): the density of unlike edges decays slowly; plot it against time.
2. Rock-paper-scissors (cyclic invasion) plus a swap reaction A|B → B|A at varying rate: coexistence with spirals at low swap rate, loss of species at high rate. Qualitative check against Reichenbach, Mobilia and Frey (2007).
3. Chiral vertex rotation (A, B, C) → (C, A, B) with and without its mirror: compare patterns.
4. Normal against shuffled runs for each test level: the size of the spatial effect.

Solver (tools/solver.py, Phase 2): enumerate block arrangements reachable by swaps (deduplicated); screen each with the mean-field equations; validate promising ones with spatial runs over several seeds; search short switch sequences with waits (for example, up to 3 switches with waits from a small set). Report per level: solvable or not, smallest number of switches found, whether timing matters, mean-field verdict against spatial verdict, seed failure rate.

Bots: switch sequence bots from the solver; a dithering bot that alternates two arrangements quickly.

Tests (pytest):
1. Lookup tables agree with brute-force matching for all colour tuples.
2. A clockwise-only vertex reaction does not fire on the mirror arrangement.
3. Cell and edge counts: swap reactions conserve colour counts exactly.
4. Shuffled control matches the mean-field ODE within 2 % on a large grid.
5. Determinism with a fixed seed and switch timeline.
6. Goal detection for extinct, hold and reach.
7. Swapping blocks conserves the multiset of blocks and respects locks.

## Levels

Level file (levels/Lxx.json):

```
{
  "id": "L04",
  "grid": { "type": "hex", "n": 64 },
  "palette": 3,
  "reactions": [
    { "shape": "edge",   "in": [0, 1], "out": [0, 0], "rate": 1.0 },
    { "shape": "edge",   "in": [1, 2], "out": [1, 1], "rate": 1.0 },
    { "shape": "edge",   "in": [2, 0], "out": [2, 2], "rate": 1.0 },
    { "shape": "vertex", "in": [0, 1, 2], "out": [2, 0, 1], "rate": 0.3, "chiral": true }
  ],
  "locked": [[0, "in", 0]],
  "start": { "seed": 7, "fractions": [0.34, 0.33, 0.33], "burn_in_sweeps": 400 },
  "goals": [ { "type": "extinct", "color": 2 } ],
  "fail": [ ],
  "meta": { "min_switches_found": 2, "timing_matters": true, "mean_field_wrong": true }
}
```

Campaign of 8–10 levels: learn the swap on a two-reaction level; voter-model coarsening; tip a balance; rock-paper-scissors: keep all three alive; rock-paper-scissors: make one extinct; a chiral level; a timing level (switch, wait, switch back); mixed. Chris picks and orders levels from solver output.

## Godot game

- Godot 4.3 or later (confirm the user's version), GDScript only, Compatibility renderer, web export with thread support off. Flag any Godot 3 idiom (yield, export without @, onready without @, PoolByteArray...).
- Simulation in plain classes (RefCounted), no Nodes, testable headless: Graph, Reactions (with lookup tables), Sim, Goals, LevelLoader. Colours in a PackedByteArray. Scenes only display state and pass input.
- Golden tests: `godot --headless --script res://tests/run_tests.gd` checks lookup tables, chirality, conservation, goal detection, and compares population curves of fixed-seed runs with the Python twin statistically (same mean-field limit in shuffled mode; same qualitative outcome in normal mode).
- Rendering: one MultiMesh or a texture-based hex renderer; colours updated per frame from the byte array. Flat cells, thin dark borders.
- UI: reaction panel with equations drawn as colour blocks (edge: two blocks and an arrow; vertex: a triangle of three blocks with a rotation arrow), a lamp beside each, click or drag to swap two blocks, locked blocks marked. Pause, speed (×1, ×4, ×16), restart, level select. Population chart with goal bands. Optional firing-flash overlay toggle.
- Palette: no fixed limit on the number of colours (Chris, 2026-11-18: find out how many make the game interesting); colours distinguishable by colour-blind players (start from the Okabe–Ito palette, recalled; check), plus a distinct glyph on each block in the equations.
- Audio: each reaction has its own soft pluck; firing rate sets the density of plucks, so the mosaic sounds like a chord that changes with the balance. AudioStreamGenerator if it behaves in the web export, otherwise CC0 samples, credited.
- Progress: completed levels in user://progress.json.
- Performance: benchmark in Phase 0. Target: at least 4 sweeps per second at ×1 on the chosen grid in Chrome on the user's laptop (guess). If not met, shrink the grid (48 × 48, then 40 × 40) before anything else.

## Repository layout

```
mosaic/
  CLAUDE.md
  README.md              # Phase 5
  LICENSE                # chosen by the user
  model/                 # Python package mosaic_model + tests
  toy/                   # pygame-ce viewer
  tools/                 # science checks, solver, bots, foam generator, golden export
  levels/  grids/        # shared JSON (levels, precomputed foams)
  game/                  # Godot project: sim/ scenes/ ui/ audio/ tests/
  docs/                  # screenshots, GIFs for the itch page
  runs/                  # gitignored
```

## Conventions

- Python 3.11+, one virtual environment at the repo root (.venv). Dependencies: numpy, pygame-ce, pytest. Ask before adding anything (numba, matplotlib, scipy for the foam generator).
- Parameters and content in JSON or TOML, never hard-coded. One seed per run.
- Commit at the end of each task with a message naming the phase and task.

## Plan

Dates and hours are guesses. If a phase runs more than a day late, apply the cut list.

Phase 0, setup and benchmark (Wed 18 – Thu 19 Nov, about 2 h)
- Public GitHub repo, layout above, .venv, pytest running.
- Godot benchmark: random sequential updates with one edge reaction on 64 × 64 hexes, web export, sweeps per second in Chrome. Report and choose the grid size.

Phase 1, Python twin and viewer (Thu 19 – Sat 21 Nov, about 5 h)
- Model, tests 1–7, science checks 1–4, minimal pygame viewer, three test levels (voter, rock-paper-scissors with swaps, chiral).
- Gate B on Sunday 22: Chris plays the viewer. Is switching live interesting? Are the lamps and chart enough to understand what happens? Record the decision in Status.

Phase 2, solver and levels (Sun 22 – Tue 24 Nov, about 4 h)
- Solver, bots, fun-hypothesis table, 12–15 candidate levels with stationary starts. Chris picks 8–10.

Phase 3, Godot core (Tue 24 – Thu 26 Nov, about 6 h)
- Graph, reactions, sim, rendering, reaction panel with lamps and swapping, pause and speed, goals, population chart, golden tests.
- Acceptance: one level playable start to finish in a web export within the performance budget.

Phase 4, game (Fri 27 – Sat 28 Nov, about 5 h)
- Level select, all chosen levels, short tutorial on level 1, palette and glyphs, audio, progress save. Foam grids only if everything else is done.

Phase 5, freeze and submit (Sun 29 – Mon 30 Nov, about 3 h; Tue 1 Dec buffer)
- Feature freeze on Sunday 29; bug fixes only after that.
- Two or three outside players try it without help; note where they get stuck.
- README (how to play, the science in one paragraph, credits), itch page text and screenshots, web build, submission on Monday 30.

Cut list, in this order: foam grids; firing-flash overlay; AudioStreamGenerator (use samples); levels beyond six; chiral reactions (keep cell and edge only).

## Decisions

- Godot Wild Jam #99 dropped; entry for GitHub Game Off 2026, theme SHIFT.
- Sand frame, phase-space archipelago and the one-button boat race (never started) dropped in favour of the reaction mosaic.
- Reactions (number, shape, rate) are given by each level; the player can only switch colour blocks between reaction slots, live, at any time.
- Large grid, random sequential updates, start from a stationary state of the level's initial arrangement.
- A lamp beside each equation shows its firing rate.
- Goals only, no score: extinct, hold, reach.
- Hexagonal grid first; precomputed foam grids only if time remains.
- The parallel with First Rain (ecology) is accepted.

## Open questions (as tasks)

- Title: search Steam, itch.io and the web for clashes before adopting "Mosaic Shift" or any other title.
- Licence for the public repo.
- Confirm the jam's rule on AI-assisted code before submitting.
- Godot version installed.
- Grid size, from the Phase 0 benchmark.
- Switch cooldown: none by default; add one if the dithering bot trivialises levels.
- Palette and glyphs: final choice after the first playable build.

## Sources

- T. Reichenbach, M. Mobilia, E. Frey, "Mobility promotes and jeopardizes biodiversity in rock-paper-scissors games", Nature 448, 1046–1049 (2007), https://doi.org/10.1038/nature06095 (seen in search results): with cyclic invasion plus mobility, species coexist in spirals below a critical mobility and are lost above it.
- Nearest games (seen in search results): HOX, https://quintin-steiner.itch.io/hox (design rules to grow a seed into a shape); Automaton, https://mestiez.itch.io/automaton (devise a ruleset to satisfy level constraints). Both deterministic and synchronous.
- Recalled, not verified in this session: voter model coarsening in two dimensions; Kawasaki exchange dynamics; Gillespie's stochastic simulation algorithm (1976); Kolmogorov's criterion for detailed balance; the Okabe–Ito colour-blind-safe palette.
