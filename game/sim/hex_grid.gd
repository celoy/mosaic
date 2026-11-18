class_name HexGrid
extends RefCounted
## n x n hexagonal grid, "odd-r" offset rows (odd rows shifted right by half a cell),
## periodic in both directions (torus). n must be even so that row parity wraps.
## Cell index i = r * n + c. Each cell owns three edges (east, south-west, south-east),
## so the edge count is exactly 3 n^2.

const SQRT3 := 1.7320508075688772

var n: int
var cell_count: int
var edge_a := PackedInt32Array()
var edge_b := PackedInt32Array()


func _init(n_: int) -> void:
	assert(n_ >= 4 and n_ % 2 == 0, "HexGrid: n must be even and >= 4")
	n = n_
	cell_count = n * n
	edge_a.resize(3 * cell_count)
	edge_b.resize(3 * cell_count)
	var k := 0
	for r in n:
		var r2 := (r + 1) % n
		for c in n:
			var i := r * n + c
			var east := r * n + (c + 1) % n
			var sw: int
			var se: int
			if r % 2 == 0:
				sw = r2 * n + (c - 1 + n) % n
				se = r2 * n + c
			else:
				sw = r2 * n + c
				se = r2 * n + (c + 1) % n
			for j in [east, sw, se]:
				edge_a[k] = i
				edge_b[k] = j
				k += 1


## Pixel centre of cell i for a hex of circumradius `size` (pointy-top).
func cell_center(i: int, size: float) -> Vector2:
	var r := i / n
	var c := i % n
	return Vector2(size * SQRT3 * (c + 0.5 * (r & 1)), size * 1.5 * r)


## Neighbour lists built from the edge arrays (used by tests).
func neighbours() -> Array:
	var nb: Array = []
	nb.resize(cell_count)
	for i in cell_count:
		nb[i] = PackedInt32Array()
	for e in edge_a.size():
		nb[edge_a[e]].append(edge_b[e])
		nb[edge_b[e]].append(edge_a[e])
	return nb
