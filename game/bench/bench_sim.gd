class_name BenchSim
extends RefCounted
## Phase 0 benchmark kernel: random sequential updates of a single edge reaction
## A|B -> C|D on a HexGrid. One attempt picks an edge uniformly at random and looks up
## its ordered colour pair in a table that already folds in both orientations.
## One sweep = one attempt per edge (3 n^2 attempts).

var grid: HexGrid
var palette: int
var colors := PackedByteArray()
var rng := RandomNumberGenerator.new()
## table[a * palette + b] = encoded outcome (a' * palette + b') or -1 if nothing matches.
var table := PackedInt32Array()
var attempts_done := 0
var fired := 0


func _init(n: int, palette_: int, seed_: int) -> void:
	grid = HexGrid.new(n)
	palette = palette_
	rng.seed = seed_
	colors.resize(grid.cell_count)
	for i in grid.cell_count:
		colors[i] = rng.randi() % palette
	table.resize(palette * palette)
	table.fill(-1)


## Install the single edge reaction a_in|b_in -> a_out|b_out (rate 1, the only rate,
## so every match is applied). Both orientations of the edge are matched.
## Assumes a_in != b_in (a symmetric input pair would need a random orientation choice).
func set_edge_reaction(a_in: int, b_in: int, a_out: int, b_out: int) -> void:
	assert(a_in != b_in)
	table.fill(-1)
	table[a_in * palette + b_in] = a_out * palette + b_out
	table[b_in * palette + a_in] = b_out * palette + a_out


func edge_count() -> int:
	return grid.edge_a.size()


## Perform k attempts. Returns the number of reactions that fired.
func run_attempts(k: int) -> int:
	var ea := grid.edge_a
	var eb := grid.edge_b
	var tab := table
	var p := palette
	var e := ea.size()
	var c := colors  # copy-on-write: written back below
	var r := rng
	var hits := 0
	for _t in k:
		var ed := r.randi() % e
		var i := ea[ed]
		var j := eb[ed]
		var out := tab[c[i] * p + c[j]]
		if out >= 0:
			c[i] = out / p
			c[j] = out % p
			hits += 1
	colors = c
	attempts_done += k
	fired += hits
	return hits


func sweeps_done() -> float:
	return float(attempts_done) / float(edge_count())


func counts() -> PackedInt32Array:
	var out := PackedInt32Array()
	out.resize(palette)
	for v in colors:
		out[v] += 1
	return out
