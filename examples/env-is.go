package main

import (
	"math"

	. "github.com/fogleman/pt/pt"
)

// makeProceduralHDRI builds an HDR environment texture entirely in memory:
// a dim sky with a small, very bright spot. This is exactly the regime in
// which uniform escape-ray sampling converges slowly and environment map
// importance sampling helps the most. A ColorTexture is built directly so
// the HDR range is not clamped to 8-bit image encoding.
func makeProceduralHDRI(w, h int) Texture {
	data := make([]Color, w*h)
	cx, cy, r := w*3/4, h/5, 3
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx := float64(x - cx)
			dy := float64(y - cy)
			v := 0.03
			if dx*dx+dy*dy <= float64(r*r) {
				v = 8.0
			}
			data[y*w+x] = Color{v, v, v}
		}
	}
	return &ColorTexture{Width: w, Height: h, Data: data}
}

func main() {
	scene := Scene{}
	scene.Texture = makeProceduralHDRI(256, 128)
	scene.TextureAngle = math.Pi / 6

	// Turn on environment map importance sampling. The distribution is
	// built once when the scene compiles and is reused by every path.
	scene.EnvSampling = true

	floor := DiffuseMaterial(HexColor(0x999999))
	scene.Add(NewCube(V(-20, -2, -20), V(20, -1, 20), floor))
	scene.Add(NewSphere(V(0, -0.2, 0), 0.8, GlossyMaterial(White, 1.5, Radians(5))))

	camera := LookAt(V(2, 1.5, 3), V(0, -0.5, 0), V(0, 1, 0), 50)
	sampler := NewSampler(4, 4)
	renderer := NewRenderer(&scene, &camera, sampler, 640, 360)
	renderer.SamplesPerPixel = 64
	renderer.Render()
	if err := SavePNG("out_env_is.png", renderer.Buffer.Image(ColorChannel)); err != nil {
		panic(err)
	}
}
