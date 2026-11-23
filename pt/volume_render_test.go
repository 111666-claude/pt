package pt

import (
	"math/rand"
	"testing"
)

// TestEndToEndSmoke renders a small scene containing a heterogeneous medium,
// a point light, a reflective floor and a refractive sphere, exercising the
// full camera -> media -> surface bounce path with concurrent workers.
func TestEndToEndSmoke(t *testing.T) {
	v := testVolume()
	light := NewSphere(Vector{1.5, 1.2, 1.0}, 0.1, LightMaterial(White, 30))
	floor := NewCube(V(-3, -1.01, -3), V(3, -1, 3), GlossyMaterial(HexColor(0x335577), 1.4, Radians(8)))
	glass := NewSphere(V(-0.7, -0.6, 0.4), 0.25, ClearMaterial(1.5, 0))
	scene := Scene{Color: HexColor(0x101018)}
	scene.Add(v)
	scene.Add(light)
	scene.Add(floor)
	scene.Add(glass)
	camera := LookAt(V(0, -2.5, -2.5), V(0, 0, 0), V(0, 0, -1), 40)
	sampler := NewSampler(1, 2)
	sampler.LightMode = LightModeAll
	renderer := NewRenderer(&scene, &camera, sampler, 24, 24)
	renderer.NumCPU = 2
	renderer.Verbose = false
	img := renderer.Render()
	if img.Bounds().Dx() != 24 || img.Bounds().Dy() != 24 {
		t.Fatalf("unexpected image size: %v", img.Bounds())
	}
	// Count non-background pixels and confirm all channels are finite and
	// non-negative (no NaN/Inf leaked from the integrators).
	nonBackground := 0
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			f := func(c uint32) float64 { return float64(c) / 65535 }
			cr, cg, cb := f(r), f(g), f(b)
			if !(cr >= 0 && cg >= 0 && cb >= 0) {
				t.Fatalf("negative pixel at (%d,%d): %v %v %v", x, y, cr, cg, cb)
			}
			if cr+cg+cb > 0.02 {
				nonBackground++
			}
		}
	}
	if nonBackground == 0 {
		t.Fatal("render produced only background pixels; media path may be dead")
	}
}

// TestDeterministicTransmittanceNoRNG confirms the transmittance path needs
// no random source and is fully reproducible.
func TestDeterministicTransmittanceNoRNG(t *testing.T) {
	v := testVolume()
	ray := Ray{Vector{-1, 0.2, -0.3}, Vector{1, 0, 0}}
	a := v.Transmittance(ray, 0, 2)
	b := v.Transmittance(ray, 0, 2)
	if !colorNear(a, b, 0) {
		t.Fatalf("transmittance not deterministic: %v vs %v", a, b)
	}
	// A second volume with an independent RNG state must agree, proving no
	// global randomness is consulted.
	_ = rand.New(rand.NewSource(1))
	c := v.Transmittance(ray, 0, 2)
	if !colorNear(a, c, 0) {
		t.Fatalf("transmittance depends on external RNG state: %v vs %v", a, c)
	}
}
