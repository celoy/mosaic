extends Node2D
## Phase 0 benchmark scene. For each grid size, runs the BenchSim kernel with a per-frame
## time budget for the simulation, renders every frame (MultiMesh of hexes coloured from
## a data texture), and reports sweeps per second and frames per second.
## User args (after `--`): --sizes=40,48,64  --seconds=6  --budget-ms=12  --quit

const PALETTE := 3
const SEED := 7
const CHUNK := 512  # attempts between clock checks
const WARMUP_S := 1.0
const BG := Color(0.12, 0.12, 0.14)

var sizes: Array[int] = [40, 48, 64, 96, 128]
var measure_s := 6.0
var budget_ms := 12.0
var quit_when_done := false

var sim: BenchSim
var size_idx := -1
var phase_start_us := 0
var measuring := false
var m_attempts0 := 0
var m_frames := 0
var m_sim_us := 0
var m_t0_us := 0
var results: Array = []

var mmi: MultiMeshInstance2D
var mat: ShaderMaterial
var img: Image
var tex: ImageTexture
var label: Label


func _ready() -> void:
	_parse_args()
	RenderingServer.set_default_clear_color(BG)
	mmi = MultiMeshInstance2D.new()
	mat = ShaderMaterial.new()
	mat.shader = load("res://bench/hex_cells.gdshader")
	mmi.material = mat
	add_child(mmi)
	label = Label.new()
	label.position = Vector2(8, 8)
	label.add_theme_color_override("font_color", Color.WHITE)
	label.add_theme_color_override("font_outline_color", Color.BLACK)
	label.add_theme_constant_override("outline_size", 6)
	add_child(label)
	print("BENCH start engine=%s os=%s sizes=%s seconds=%s budget_ms=%s" % [
		Engine.get_version_info().string, OS.get_name(), sizes, measure_s, budget_ms])
	_next_size()


func _parse_args() -> void:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--sizes="):
			sizes.clear()
			for s in a.trim_prefix("--sizes=").split(","):
				sizes.append(int(s))
		elif a.begins_with("--seconds="):
			measure_s = float(a.trim_prefix("--seconds="))
		elif a.begins_with("--budget-ms="):
			budget_ms = float(a.trim_prefix("--budget-ms="))
		elif a == "--quit":
			quit_when_done = true


func _next_size() -> void:
	size_idx += 1
	if size_idx >= sizes.size():
		_finish()
		return
	var n := sizes[size_idx]
	sim = BenchSim.new(n, PALETTE, SEED)
	sim.set_edge_reaction(0, 1, 0, 0)  # A|B -> A|A, rate 1 (one-way voter / invasion)
	_build_mesh(n)
	phase_start_us = Time.get_ticks_usec()
	measuring = false


func _build_mesh(n: int) -> void:
	var view := get_viewport_rect().size
	# Fit the n x n hex block: width ~ sqrt3*(n+0.5)*s, height ~ (1.5n+0.5)*s.
	var s: float = min((view.x - 20.0) / (HexGrid.SQRT3 * (n + 0.5)),
		(view.y - 60.0) / (1.5 * n + 0.5))
	var hex_r := s * 0.92  # gap between hexes shows the dark background as borders
	var verts := PackedVector2Array()
	for k in 6:
		var a0 := deg_to_rad(60.0 * k - 30.0)
		var a1 := deg_to_rad(60.0 * (k + 1) - 30.0)
		verts.append_array([Vector2.ZERO, hex_r * Vector2(cos(a0), sin(a0)),
			hex_r * Vector2(cos(a1), sin(a1))])
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_2D
	mm.mesh = mesh
	mm.instance_count = sim.grid.cell_count
	var origin := Vector2(10.0 + s * HexGrid.SQRT3 * 0.5, 50.0 + s)
	for i in sim.grid.cell_count:
		mm.set_instance_transform_2d(i, Transform2D(0.0, origin + sim.grid.cell_center(i, s)))
	mmi.multimesh = mm
	img = Image.create_from_data(n, n, false, Image.FORMAT_R8, sim.colors)
	tex = ImageTexture.create_from_image(img)
	mat.set_shader_parameter("cells", tex)
	mat.set_shader_parameter("n", n)


func _process(_delta: float) -> void:
	if sim == null:
		return
	var now := Time.get_ticks_usec()
	if not measuring and now - phase_start_us > WARMUP_S * 1e6:
		measuring = true
		m_t0_us = now
		m_attempts0 = sim.attempts_done
		m_frames = 0
		m_sim_us = 0
	# Simulation under a per-frame time budget.
	var t0 := Time.get_ticks_usec()
	var limit := t0 + int(budget_ms * 1000.0)
	while Time.get_ticks_usec() < limit:
		sim.run_attempts(CHUNK)
	var t1 := Time.get_ticks_usec()
	# Upload colours (one call, no per-cell loop).
	img.set_data(sim.grid.n, sim.grid.n, false, Image.FORMAT_R8, sim.colors)
	tex.update(img)
	if measuring:
		m_frames += 1
		m_sim_us += t1 - t0
		var el := (t1 - m_t0_us) * 1e-6
		var sw := float(sim.attempts_done - m_attempts0) / sim.edge_count()
		label.text = "n=%d  edges=%d  sweeps/s=%.1f  fps=%.1f  us/attempt=%.3f" % [
			sim.grid.n, sim.edge_count(), sw / max(el, 1e-6), m_frames / max(el, 1e-6),
			m_sim_us / max(1.0, float(sim.attempts_done - m_attempts0))]
		if el >= measure_s:
			_report(el)
			_next_size()
	else:
		label.text = "n=%d warming up..." % sim.grid.n


func _report(el: float) -> void:
	var att := sim.attempts_done - m_attempts0
	var r := {
		"n": sim.grid.n,
		"edges": sim.edge_count(),
		"seconds": snappedf(el, 0.001),
		"sweeps_per_s": snappedf(att / float(sim.edge_count()) / el, 0.01),
		"fps": snappedf(m_frames / el, 0.1),
		"us_per_attempt": snappedf(m_sim_us / float(att), 0.0001),
		"sim_fraction": snappedf(m_sim_us * 1e-6 / el, 0.001),
		"counts": Array(sim.counts()),
	}
	results.append(r)
	print("BENCH ", JSON.stringify(r))


func _finish() -> void:
	sim = null
	var lines := PackedStringArray(["Benchmark done (budget %.0f ms/frame):" % budget_ms])
	for r in results:
		lines.append("n=%d  sweeps/s=%.1f  fps=%.1f  us/attempt=%.3f" % [
			r.n, r.sweeps_per_s, r.fps, r.us_per_attempt])
	label.text = "\n".join(lines)
	print("BENCH_DONE ", JSON.stringify(results))
	if quit_when_done:
		get_tree().quit()
