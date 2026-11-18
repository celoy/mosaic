shader_type canvas_item;
// Colours each MultiMesh instance (one hex per cell) from an n x n R8 texture that
// holds the cell colour indices. Instance id == cell index.

uniform sampler2D cells : filter_nearest;
uniform int n = 64;
uniform vec4 pal0 : source_color = vec4(0.902, 0.624, 0.0, 1.0);    // Okabe-Ito orange
uniform vec4 pal1 : source_color = vec4(0.337, 0.706, 0.914, 1.0);  // sky blue
uniform vec4 pal2 : source_color = vec4(0.0, 0.620, 0.451, 1.0);    // bluish green
uniform vec4 pal3 : source_color = vec4(0.835, 0.369, 0.0, 1.0);    // vermillion

void vertex() {
	int id = INSTANCE_ID;
	float v = texelFetch(cells, ivec2(id % n, id / n), 0).r;
	int k = int(round(v * 255.0));
	COLOR = k == 0 ? pal0 : (k == 1 ? pal1 : (k == 2 ? pal2 : pal3));
}
