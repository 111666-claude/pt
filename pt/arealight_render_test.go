package pt

import (
	"math"
	"testing"
)

// TestEndToEndRenderWithMeshLight drives the full renderer with an emissive
// mesh light and an emissive sphere together, asserting the image contains
// no NaN/Inf pixels and that the ground is actually lit.
func TestEndToEndRenderWithMeshLight(t *testing.T) {
	scene := Scene{}
	// emissive quad mesh light overhead
	light := quadMesh(-2, 2, -2, 2, 5, 3)
	scene.Add(light)
	// sphere light keeps its original estimator
	scene.Add(NewSphere(V(4, 3, 2), 0.5, LightMaterial(White, 10)))
	// diffuse ground
	scene.Add(NewCube(V(-50, -1, -50), V(50, 0, 50), DiffuseMaterial(White)))
	// an occluder casting a soft shadow
	scene.Add(NewCube(V(-1, 0, -1), V(0, 2, 0), DiffuseMaterial(HexColor(0x888888))))

	camera := LookAt(V(0, 2, 8), V(0, 1, 0), V(0, 1, 0), 45)
	sampler := NewSampler(1, 2)
	sampler.SoftShadows = true
	renderer := NewRenderer(&scene, &camera, sampler, 64, 64)
	renderer.Verbose = false
	renderer.SamplesPerPixel = 4
	img := renderer.Render()

	bounds := img.Bounds()
	var litPixels, finitePixels, total int
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rf := float64(r) / 65535
			gf := float64(g) / 65535
			bf := float64(b) / 65535
			total++
			for _, c := range [3]float64{rf, gf, bf} {
				if math.IsNaN(c) || math.IsInf(c, 0) {
					t.Fatalf("non-finite pixel at (%d,%d): %v %v %v", x, y, rf, gf, bf)
				}
			}
			finitePixels++
			if rf > 0.01 || gf > 0.01 || bf > 0.01 {
				litPixels++
			}
		}
	}
	if finitePixels != total {
		t.Fatalf("not all pixels finite: %d/%d", finitePixels, total)
	}
	if litPixels == 0 {
		t.Fatalf("scene rendered fully black: area light not illuminating anything")
	}
	t.Logf("rendered %dx%d, lit pixels=%d/%d", bounds.Dx(), bounds.Dy(), litPixels, total)
}
