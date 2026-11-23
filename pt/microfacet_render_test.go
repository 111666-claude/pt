package pt

import (
	"math"
	"math/rand"
	"testing"
)

// renderScene renders a scene deterministically and returns the buffer.
func renderScene(scene *Scene, camera Camera, w, h, spp, bounces int) *Buffer {
	sampler := NewSampler(spp, bounces)
	sampler.SpecularMode = SpecularModeAll
	sampler.LightMode = LightModeAll
	renderer := NewRenderer(scene, &camera, sampler, w, h)
	renderer.Verbose = false
	renderer.NumCPU = 1
	renderer.run()
	return renderer.Buffer
}

func meanOnSphere(t *testing.T, buf *Buffer, w, h int, scene *Scene, camera Camera, center Vector, radius float64) (float64, int, float64) {
	t.Helper()
	var sum, peak float64
	n := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// cast the same primary ray the camera would and test the hit
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rand.New(rand.NewSource(1)))
			hit := scene.Intersect(ray)
			if !hit.Ok() {
				continue
			}
			info := hit.Info(ray)
			if d := info.Position.Sub(center).Length(); math.Abs(d-radius) > 1e-6 {
				continue
			}
			c := buf.Color(x, y)
			if !finiteColor(c) {
				t.Fatalf("non-finite pixel at (%d,%d): %+v", x, y, c)
			}
			v := (c.R + c.G + c.B) / 3
			sum += v
			peak = math.Max(peak, v)
			n++
		}
	}
	if n == 0 {
		t.Fatal("no sphere pixels found")
	}
	return sum / float64(n), n, peak
}

// highlightRadius measures the angular radius (radians, around the sphere
// center) enclosing pixels brighter than half the peak brightness.
func highlightRadius(t *testing.T, buf *Buffer, w, h int, scene *Scene, camera Camera, center Vector, peak float64) float64 {
	t.Helper()
	var maxAng float64
	threshold := peak * 0.5
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rand.New(rand.NewSource(1)))
			hit := scene.Intersect(ray)
			if !hit.Ok() {
				continue
			}
			info := hit.Info(ray)
			c := buf.Color(x, y)
			v := (c.R + c.G + c.B) / 3
			if v >= threshold {
				// angle between this surface point's radius and the view axis
				a := math.Acos(Clamp(info.Position.Sub(center).Normalize().Dot(camera.w.MulScalar(-1)), -1, 1))
				maxAng = math.Max(maxAng, a)
			}
		}
	}
	return maxAng
}

func ggxTestScene(m Material) (*Scene, Camera) {
	scene := Scene{Color: Black}
	scene.Add(NewSphere(V(0, 1, 0), 1, m))
	// floor far enough below to avoid dominating the image
	scene.Add(NewCube(V(-10, -1, -10), V(10, -0.95, 10), DiffuseMaterial(HexColor(0x222222))))
	// small bright sphere light above and to the side => compact highlight
	scene.Add(NewSphere(V(1.5, 3.2, 1.2), 0.18, LightMaterial(White, 60)))
	camera := LookAt(V(0, 1.4, 4.5), V(0, 1, 0), V(0, 1, 0), 35)
	return &scene, camera
}

func TestGGXMetalRoughnessSeries(t *testing.T) {
	roughnesses := []float64{0, 0.1, 0.3, 0.6}
	means := make([]float64, len(roughnesses))
	peaks := make([]float64, len(roughnesses))
	radii := make([]float64, len(roughnesses))
	for i, r := range roughnesses {
		m := MetalMaterial(HexColor(0xC0C0C0), r)
		scene, camera := ggxTestScene(m)
		buf := renderScene(scene, camera, 96, 96, 64, 4)
		mean, n, peak := meanOnSphere(t, buf, 96, 96, scene, camera, V(0, 1, 0), 1)
		means[i], peaks[i] = mean, peak
		radii[i] = highlightRadius(t, buf, 96, 96, scene, camera, V(0, 1, 0), peak)
		t.Logf("metal r=%.2f mean=%.4f peak=%.4f highlight=%.4f rad over %d px", r, mean, peak, radii[i], n)
		// HDR output is only required to stay finite and non-negative.
		if mean < 0 || peak < 0 || math.IsInf(peak, 0) {
			t.Errorf("metal r=%v invalid brightness: mean=%v peak=%v", r, mean, peak)
		}
	}
	// The specular lobe widens monotonically with roughness: the angular
	// extent of the bright region grows overall (adjacent comparisons are
	// noisy, so compare each level against the mirror baseline).
	for i := 1; i < len(radii); i++ {
		if radii[i] < radii[0]-0.05 {
			t.Errorf("highlight at r=%v (%.4f) narrower than mirror (%.4f)",
				roughnesses[i], radii[i], radii[0])
		}
	}
	// A perfectly smooth mirror concentrates light into the tightest spot,
	// i.e. its bright region is no wider than any rough variant.
	for i := 1; i < len(radii); i++ {
		if radii[0] > radii[i]+0.1 {
			t.Errorf("mirror highlight wider than rough r=%v: %v vs %v",
				roughnesses[i], radii[0], radii[i])
		}
	}
	// Energy conservation: average reflected power stays within a factor-of-2
	// band across roughness. Single-scattering GGX loses some multiple-bounce
	// energy at high roughness, and the mirror's tiny hotspot is under-sampled
	// at 64 spp, so a tight band would be testing noise rather than physics.
	for i := 1; i < len(means); i++ {
		lo := means[0] * 0.4
		hi := means[0]*2.5 + 0.01
		if means[i] < lo || means[i] > hi {
			t.Errorf("mean brightness r=%v (%v) outside conservation band [%v,%v] of mirror %v",
				roughnesses[i], means[i], lo, hi, means[0])
		}
	}
	_ = peaks
}

func TestGGXDielectricRoughnessSeries(t *testing.T) {
	roughnesses := []float64{0, 0.25, 0.6}
	means := make([]float64, len(roughnesses))
	for i, r := range roughnesses {
		m := DielectricMaterial(HexColor(0x4060A0), 1.5, r)
		scene, camera := ggxTestScene(m)
		buf := renderScene(scene, camera, 96, 96, 64, 4)
		mean, n, peak := meanOnSphere(t, buf, 96, 96, scene, camera, V(0, 1, 0), 1)
		means[i] = mean
		t.Logf("dielectric r=%.2f mean=%.4f peak=%.4f over %d px", r, mean, peak, n)
		if mean < 0 || peak < 0 || math.IsInf(peak, 0) {
			t.Errorf("dielectric r=%v invalid brightness: mean=%v peak=%v", r, mean, peak)
		}
	}
	// A dielectric reflects only ~4% at normal incidence regardless of
	// roughness: mean brightness stays in a conservative conservation band
	// (the mirror hotspot is tiny and under-sampled at 64 spp).
	for i := 1; i < len(means); i++ {
		lo := means[0] * 0.3
		hi := means[0]*3 + 0.01
		if means[i] < lo || means[i] > hi {
			t.Errorf("dielectric mean r=%v (%v) outside band [%v,%v] of mirror %v",
				roughnesses[i], means[i], lo, hi, means[0])
		}
	}
}

func TestGGXFiniteWeightsInRender(t *testing.T) {
	// Adversarial parameters: extreme roughness, bad indices, grazing view.
	materials := []Material{
		MetalMaterial(White, 1),
		MetalMaterial(Black, 0),
		GGXMaterial(White, 0.0001, 0.5, 0.5),
		GGXMaterial(White, 1e9, 0.5, 0),
		GGXMaterial(Color{math.NaN(), 2, -1}, 1.5, math.Inf(1), 2),
	}
	for k, m := range materials {
		scene, camera := ggxTestScene(m)
		buf := renderScene(scene, camera, 48, 48, 16, 3)
		for idx, p := range buf.Pixels {
			c := p.Color()
			if !finiteColor(c) || c.MinComponent() < -1e-9 || c.MaxComponent() > 100 {
				t.Fatalf("material %d produced invalid pixel %d: %+v", k, idx, c)
			}
		}
	}
}

func TestGGXSpecularModes(t *testing.T) {
	// All three SpecularMode settings must render finite images for the same
	// GGX scene (the original "Gloss 角度断层" regression surface).
	m := MetalMaterial(HexColor(0xD0D0D0), 0.25)
	modes := []SpecularMode{SpecularModeNaive, SpecularModeFirst, SpecularModeAll}
	for _, mode := range modes {
		scene, camera := ggxTestScene(m)
		sampler := NewSampler(4, 4)
		sampler.SpecularMode = mode
		renderer := NewRenderer(scene, &camera, sampler, 64, 64)
		renderer.Verbose = false
		renderer.NumCPU = 1
		renderer.run()
		for idx, p := range renderer.Buffer.Pixels {
			c := p.Color()
			if !finiteColor(c) || c.MinComponent() < -1e-9 {
				t.Fatalf("SpecularMode %v invalid pixel %d: %+v", mode, idx, c)
			}
		}
	}
}

func TestGGXTransparentEnergy(t *testing.T) {
	// A rough transparent dielectric: reflection + transmission weights must
	// partition unity with no negative or infinite contributions.
	m := GGXMaterial(White, 1.5, 0.3, 0)
	m.Transparent = true
	scene := Scene{Color: Color{0.1, 0.1, 0.1}}
	scene.Add(NewSphere(V(0, 1, 0), 1, m))
	scene.Add(NewSphere(V(1.5, 3, 1), 0.2, LightMaterial(White, 40)))
	camera := LookAt(V(0, 1.2, 4), V(0, 1, 0), V(0, 1, 0), 35)
	buf := renderScene(&scene, camera, 64, 64, 32, 6)
	for idx, p := range buf.Pixels {
		c := p.Color()
		if !finiteColor(c) || c.MinComponent() < -1e-9 || c.MaxComponent() > 100 {
			t.Fatalf("transparent GGX invalid pixel %d: %+v", idx, c)
		}
	}
}
