extends SceneTree
## Native headless baseline (simulation only, no rendering) plus grid sanity checks.
## godot --headless --path game --script res://tests/bench_cli.gd

const SIZES: Array[int] = [40, 48, 64, 96, 128]
const SECONDS := 2.0


func _init() -> void:
	var ok := _check_grid(8) and _check_grid(64)
	# "invade": A|B -> A|A (dies out quickly, so mostly lookup misses);
	# "exchange": A|B -> B|A (fires forever, hit path every time a 0|1 edge is drawn).
	for kind in ["invade", "exchange"]:
		for n in SIZES:
			ok = _run(kind, n) and ok
	print("BENCH_CLI ", "PASS" if ok else "FAIL")
	quit(0 if ok else 1)


func _run(kind: String, n: int) -> bool:
	var ok := true
	if true:
		var sim := BenchSim.new(n, 3, 7)
		if kind == "invade":
			sim.set_edge_reaction(0, 1, 0, 0)
		else:
			sim.set_edge_reaction(0, 1, 1, 0)
		sim.run_attempts(10000)  # warm-up
		var a0 := sim.attempts_done
		var f0 := sim.fired
		var t0 := Time.get_ticks_usec()
		while Time.get_ticks_usec() - t0 < SECONDS * 1e6:
			sim.run_attempts(4096)
		var el := (Time.get_ticks_usec() - t0) * 1e-6
		var att := sim.attempts_done - a0
		var c := sim.counts()
		var total := 0
		for v in c:
			total += v
		if total != n * n:
			push_error("cell count changed: %d != %d" % [total, n * n])
			ok = false
		print("BENCH_NATIVE ", JSON.stringify({
			"kind": kind, "n": n, "fired_fraction": snappedf((sim.fired - f0) / float(att), 0.001), "edges": sim.edge_count(),
			"sweeps_per_s": snappedf(att / float(sim.edge_count()) / el, 0.01),
			"us_per_attempt": snappedf(el * 1e6 / att, 0.0001), "counts": Array(c)}))
	return ok


## Every cell has 6 distinct neighbours, the relation is symmetric, edges = 3 n^2.
func _check_grid(n: int) -> bool:
	var g := HexGrid.new(n)
	if g.edge_a.size() != 3 * n * n:
		push_error("edge count")
		return false
	var nb := g.neighbours()
	for i in g.cell_count:
		var s := {}
		for j in nb[i]:
			s[j] = true
			if not (i in nb[j]):
				push_error("asymmetric neighbours %d %d" % [i, j])
				return false
		if s.size() != 6 or s.has(i):
			push_error("cell %d has %d distinct neighbours" % [i, s.size()])
			return false
	print("grid n=%d OK: %d cells, %d edges, 6 neighbours each" % [n, g.cell_count, g.edge_a.size()])
	return true
