extends SceneTree
## Rejection vs active-set random sequential updates, edge reactions only, in GDScript.
## godot --headless --path game --script res://tests/bench_active.gd
## Prints ms of wall time per unit of sim time for each algorithm and scenario.

var n: int
var P: int
var E: int
var ea := PackedInt32Array()
var eb := PackedInt32Array()
var inc := PackedInt32Array()  # 6 edges per cell
var col := PackedByteArray()
# lookup table on edge code a*P+b
var tot := PackedFloat32Array()
var ch_start := PackedInt32Array()
var ch_count := PackedInt32Array()
var ch_cum := PackedFloat32Array()
var ch_new := PackedInt32Array()  # encoded a'*P+b'
var K := 0.0
var rng := RandomNumberGenerator.new()


func _init() -> void:
	var rps := [[0, 1, 0, 0, 1.0], [1, 2, 1, 1, 1.0], [2, 0, 2, 2, 1.0]]
	var rps_x := rps + [[0, 1, 1, 0, 2.0]]
	var front := [[0, 1, 0, 0, 1.0]]
	for size in [64, 128]:
		for sc in [["RPS (T03 edges)", rps, 3], ["RPS + exchange (T02)", rps_x, 3],
				["front", front, 2]]:
			_scenario(size, sc[0], sc[1], sc[2])
	quit()


func _scenario(size: int, name: String, reactions: Array, palette: int) -> void:
	_grid(size)
	P = palette
	_tables(reactions)
	rng.seed = 11
	col.resize(n * n)
	var t_meas := 10.0
	if name == "front":
		var w := HexGrid.SQRT3 * n
		var g := HexGrid.new(n)
		for i in n * n:
			col[i] = 0 if g.cell_center(i, 1.0).x < 0.25 * w else 1
		t_meas = 0.1 * n
	else:
		for i in n * n:
			col[i] = rng.randi() % P
		_run_rejection(50.0)  # burn-in
	var start := col.duplicate()
	var res := {}
	for algo in ["rejection", "active"]:
		col = start.duplicate()
		rng.seed = 5
		var t0 := Time.get_ticks_usec()
		var info: Array
		if algo == "rejection":
			info = _run_rejection(t_meas)
		else:
			info = _run_active(t_meas)
		var ms := (Time.get_ticks_usec() - t0) * 1e-3 / t_meas
		res[algo] = ms
		print("BENCH_ACTIVE ", JSON.stringify({"n": n, "scenario": name, "algo": algo,
			"ms_per_t": snappedf(ms, 0.01), "fire_frac": snappedf(info[0], 0.0001),
			"active_frac": snappedf(info[1], 0.0001)}))
	print("  -> %s n=%d speed-up of active over rejection: %.2f" % [
		name, n, res["rejection"] / res["active"]])


func _grid(size: int) -> void:
	n = size
	var g := HexGrid.new(n)
	ea = g.edge_a
	eb = g.edge_b
	E = ea.size()
	inc.resize(6 * n * n)
	var fill := PackedInt32Array()
	fill.resize(n * n)
	for e in E:
		var a := ea[e]
		var b := eb[e]
		inc[6 * a + fill[a]] = e
		fill[a] += 1
		inc[6 * b + fill[b]] = e
		fill[b] += 1


func _tables(reactions: Array) -> void:
	tot.resize(P * P)
	ch_start.resize(P * P)
	ch_count.resize(P * P)
	ch_cum.clear()
	ch_new.clear()
	K = 0.0
	for a in P:
		for b in P:
			var k := a * P + b
			ch_start[k] = ch_cum.size()
			var s := 0.0
			for r in reactions:
				if a == r[0] and b == r[1]:  # stored orientation
					s += r[4]
					ch_cum.append(s)
					ch_new.append(r[2] * P + r[3])
				if b == r[0] and a == r[1]:  # reversed orientation
					s += r[4]
					ch_cum.append(s)
					ch_new.append(r[3] * P + r[2])
			tot[k] = s
			ch_count[k] = ch_cum.size() - ch_start[k]
			K = max(K, s)


## Returns [fraction of attempts that fired, mean active fraction (not tracked: -1)].
func _run_rejection(t_run: float) -> Array:
	var c := col
	var dt := 1.0 / (K * E)
	var n_att := int(t_run / dt)
	var fired := 0
	for _i in n_att:
		var e := rng.randi() % E
		var a := ea[e]
		var b := eb[e]
		var k := c[a] * P + c[b]
		var u := rng.randf() * K
		if u < tot[k]:
			var j := ch_start[k]
			while u >= ch_cum[j]:
				j += 1
			var nw := ch_new[j]
			c[a] = nw / P
			c[b] = nw % P
			fired += 1
	col = c
	return [float(fired) / max(1, n_att), -1.0]


func _run_active(t_run: float) -> Array:
	var c := col
	var act := PackedInt32Array()
	act.resize(E)
	var pos := PackedInt32Array()
	pos.resize(E)
	var m := 0
	for e in E:
		if tot[c[ea[e]] * P + c[eb[e]]] > 0.0:
			act[m] = e
			pos[e] = m
			m += 1
		else:
			pos[e] = -1
	var t := 0.0
	var att := 0
	var fired := 0
	var m_sum := 0.0
	while t < t_run and m > 0:
		t += 1.0 / (K * m)
		att += 1
		m_sum += m
		var e := act[rng.randi() % m]
		var a := ea[e]
		var b := eb[e]
		var k := c[a] * P + c[b]
		var u := rng.randf() * K
		if u >= tot[k]:
			continue
		var j := ch_start[k]
		while u >= ch_cum[j]:
			j += 1
		var nw := ch_new[j]
		var na := nw / P
		var nb := nw % P
		fired += 1
		for side in 2:
			var cell := a if side == 0 else b
			var v := na if side == 0 else nb
			if c[cell] == v:
				continue
			c[cell] = v
			for q in 6:
				var e2 := inc[6 * cell + q]
				var on := tot[c[ea[e2]] * P + c[eb[e2]]] > 0.0
				var i2 := pos[e2]
				if on and i2 < 0:
					act[m] = e2
					pos[e2] = m
					m += 1
				elif not on and i2 >= 0:
					m -= 1
					var last := act[m]
					act[i2] = last
					pos[last] = i2
					pos[e2] = -1
	col = c
	return [float(fired) / max(1, att), m_sum / max(1, att) / E]
