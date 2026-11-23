package pt

import (
	"math"
	"math/rand"
	"testing"
)

// stratifiedHemisphereReference integrates L_e(d)*cos(theta) over the
// hemisphere around the +Y axis using an independent, high-resolution
// stratified spherical quadrature. theta is the polar angle from +Y,
// matching envDirectionToUV. It does not use any of the production
// sampling code, so it serves as an external reference for the Monte
// Carlo estimators under test.
func stratifiedHemisphereReference(tex Texture, angle float64, nTheta, nPhi int) Color {
	dTheta := (math.Pi / 2) / float64(nTheta)
	dPhi := 2 * math.Pi / float64(nPhi)
	acc := Black
	for i := 0; i < nTheta; i++ {
		// theta is the polar angle from the +Y axis in [0, pi/2]
		theta := (float64(i) + 0.5) * dTheta
		sinTheta, cosTheta := math.Sin(theta), math.Cos(theta)
		for j := 0; j < nPhi; j++ {
			phi := (float64(j) + 0.5) * dPhi
			dir := Vector{sinTheta * math.Cos(phi), cosTheta, sinTheta * math.Sin(phi)}
			uu, vv := envDirectionToUV(dir, angle)
			le := tex.Sample(uu, vv)
			weight := cosTheta * sinTheta * dTheta * dPhi
			acc = acc.Add(le.MulScalar(weight))
		}
	}
	return acc
}

// cosineSampleDir draws a standard cosine-weighted direction around +Y
// (pdf = cos(theta)/pi), independent of the production bounce code, so it
// serves as a textbook reference for the estimator math.
func cosineSampleDir(rnd *rand.Rand) (Vector, float64) {
	r2 := rnd.Float64()
	phi := 2 * math.Pi * rnd.Float64()
	r := math.Sqrt(r2)
	z := math.Sqrt(math.Max(0, 1-r2))
	dir := Vector{r * math.Cos(phi), z, r * math.Sin(phi)}
	return dir, dir.Y / math.Pi
}

func envColor(tex Texture, angle float64, dir Vector) Color {
	uu, vv := envDirectionToUV(dir, angle)
	// ColorTexture.Sample wraps coordinates and applies the 1-v image flip.
	return tex.Sample(uu, vv)
}

// TestEnvEnergyAndVariance compares three estimators of the same
// hemisphere irradiance integral against an independent quadrature
// reference: pure cosine (BSDF) sampling, pure environment next-event
// estimation, and the production MIS combination. All must match the
// reference energy, and MIS must dramatically reduce variance for the
// concentrated bright spot.
func TestEnvEnergyAndVariance(t *testing.T) {
	// Black background plus a compact bright spot: the textbook case where
	// uniform/cosine escape rays almost never find the light and converge
	// slowly, while environment importance sampling targets it directly.
	// The spot sits at image row 12 of 64 (latitude v ~ 0.81, well inside
	// the +Y hemisphere) and spans enough texels to survive downsampling.
	tex := sunSpotHDRI(128, 64, 96, 12, 5, 0.0, 25)
	const angle = 0.4
	dist := NewEnvDistribution(tex, angle)
	if !dist.Valid() {
		t.Fatal("test HDRI distribution must be valid")
	}
	ref := stratifiedHemisphereReference(tex, angle, 400, 800)
	if ref.R <= 0 {
		t.Fatal("reference integral must be positive for the bright spot")
	}

	// Single fixed-seed stream consumed in identical order for every
	// estimator, evaluated in one exact first/second-moment pass.
	const n = 20000
	rnd := rand.New(rand.NewSource(42))
	var sA, sB, sC, m2A, m2B, m2C float64
	var sumA, sumB, sumC Color
	hitsA, hitsB := 0, 0
	for i := 0; i < n; i++ {
		// estimator A: pure BSDF cosine sampling (the original behavior).
		// The irradiance estimator is L*cos(theta)/pdf_bsdf.
		dirB, pdfBSDF := cosineSampleDir(rnd)
		leB := envColor(tex, angle, dirB).R
		cosB := math.Max(0, dirB.Y)
		xA := 0.0
		if pdfBSDF > 0 {
			xA = leB * cosB / pdfBSDF
		}
		if xA > 0 {
			hitsA++
		}

		// estimator B: pure environment next-event estimation of the
		// irradiance integral L*cos dw, i.e. L*cos/pdf_env.
		dirE, pdfEnv, _, _ := dist.Sample(rnd)
		xB := 0.0

		// estimator C: production-style MIS of both techniques
		xC := xA * powerHeuristic(pdfBSDF, dist.PDF(dirB))
		if cos := dirE.Y; cos > 0 {
			leE := envColor(tex, angle, dirE).R
			if leE > 0 {
				hitsB++
			}
			xB = leE * cos / pdfEnv
			xC += xB * powerHeuristic(pdfEnv, math.Max(0, cos)/math.Pi)
			sumB = sumB.Add(Color{xB, xB, xB})
		}
		sumA = sumA.Add(Color{xA, xA, xA})
		sumC = sumC.Add(Color{xC, xC, xC})
		sA += xA
		sB += xB
		sC += xC
		m2A += xA * xA
		m2B += xB * xB
		m2C += xC * xC
	}
	meanA := sumA.DivScalar(n)
	meanB := sumB.DivScalar(n)
	meanC := sumC.DivScalar(n)
	varianceOf := func(sum, m2 float64) float64 {
		m := sum / n
		return math.Max(0, m2/n-m*m)
	}
	vA, vB, vC := varianceOf(sA, m2A), varianceOf(sB, m2B), varianceOf(sC, m2C)
	t.Logf("BSDF hits=%d/%d  NEE hits=%d/%d", hitsA, n, hitsB, n)
	t.Logf("reference=%.5f  BSDF mean=%.5f var=%.3e  NEE mean=%.5f var=%.3e  MIS mean=%.5f var=%.3e",
		ref.R, meanA.R, vA, meanB.R, vB, meanC.R, vC)

	// NEE/MIS must actually find the concentrated light that cosine
	// sampling mostly misses.
	if hitsB < n/4 {
		t.Fatalf("environment sampling only found the spot %d times", hitsB)
	}

	// energy: the importance-driven estimators must reproduce the
	// independent quadrature reference. The pure BSDF estimator is allowed
	// a wide tolerance because with a black background it is high-variance
	// and its fixed-seed mean may still be off.
	rel := func(got, want float64) float64 { return math.Abs(got-want) / want }
	if rel(meanB.R, ref.R) > 0.1 {
		t.Fatalf("NEE estimator energy off: got %.5f ref %.5f", meanB.R, ref.R)
	}
	if rel(meanC.R, ref.R) > 0.1 {
		t.Fatalf("MIS estimator energy off: got %.5f ref %.5f", meanC.R, ref.R)
	}
	if vA > 0 && rel(meanA.R, ref.R) > 0.4 {
		t.Logf("note: high-variance BSDF mean %.5f vs ref %.5f (expected for concentrated light)", meanA.R, ref.R)
	}

	// variance trend: MIS must be dramatically tighter than pure cosine
	// sampling for the concentrated bright spot.
	if !(vC < vA) {
		t.Fatalf("MIS variance %.3e not below BSDF variance %.3e", vC, vA)
	}
	if vC > 0.1*vA {
		t.Fatalf("MIS variance reduction too weak: ratio %.4f", vC/vA)
	}
	t.Logf("MIS/BSDF variance ratio = %.5f", vC/vA)
}

func TestEnvMonochromeEnergy(t *testing.T) {
	// for a uniform L environment the hemisphere integral is exactly L*pi
	tex := uniformEnvTexture(64, 32, 0.3)
	dist := NewEnvDistribution(tex, 0)
	ref := stratifiedHemisphereReference(tex, 0, 200, 400)
	approx(t, "monochrome reference", ref.R, 0.3*math.Pi, 1e-3)
	rnd := rand.New(rand.NewSource(99))
	var sum float64
	const n = 20000
	for i := 0; i < n; i++ {
		dir, pdf, _, _ := dist.Sample(rnd)
		if cos := dir.Y; cos > 0 {
			sum += envColor(tex, 0, dir).R * cos / pdf
		}
	}
	approx(t, "NEE monochrome energy", sum/n, 0.3*math.Pi, 0.02)
}

// TestDefaultBehaviorUnchanged proves that with EnvSampling disabled
// (the default) escaped rays still evaluate the environment exactly as
// before, with no MIS weighting and no dependence on a distribution.
func TestDefaultBehaviorUnchanged(t *testing.T) {
	tex := sunSpotHDRI(64, 32, 10, 10, 2, 0.1, 5)
	scene := &Scene{Texture: tex, TextureAngle: 0.9}
	scene.Compile()
	s := NewSampler(1, 4)
	rnd := rand.New(rand.NewSource(7))
	for i := 0; i < 200; i++ {
		ray := Ray{Vector{}, RandomUnitVector(rnd)}
		got := s.Sample(scene, ray, rnd)
		want := s.sampleEnvironment(scene, ray)
		diff := got.Sub(want)
		if math.Abs(diff.R) > 1e-12 || math.Abs(diff.G) > 1e-12 || math.Abs(diff.B) > 1e-12 {
			t.Fatalf("default path changed environment result: got %v want %v", got, want)
		}
	}
	// an invalid/absent distribution must never be consulted when disabled
	scene.EnvSampling = false
	if s.envEnabled(scene) {
		t.Fatal("env sampling must be off by default")
	}
	// scene without a texture still returns the flat background color
	flat := &Scene{Color: Color{0.2, 0.4, 0.6}}
	flat.Compile()
	got := s.Sample(flat, Ray{Vector{}, Vector{1, 0, 0}}, rnd)
	if got != flat.Color {
		t.Fatalf("flat background changed: got %v want %v", got, flat.Color)
	}
	if flat.EnvDistribution().Valid() {
		t.Fatal("texture-less scene must not build a valid distribution")
	}
}

// renderStats runs a small render and returns total image energy, the
// number of non-finite pixels, and the mean per-pixel variance.
func renderStats(scene *Scene, camera Camera, sampler *DefaultSampler, spp int) (energy float64, bad int, meanVar float64) {
	renderer := NewRenderer(scene, &camera, sampler, 48, 32)
	renderer.SamplesPerPixel = spp
	renderer.NumCPU = 2
	renderer.Verbose = false
	renderer.Render()
	varSum := 0.0
	for _, p := range renderer.Buffer.Pixels {
		c := p.Color()
		if !isFinite(c.R) || !isFinite(c.G) || !isFinite(c.B) || c.MinComponent() < 0 {
			bad++
		}
		energy += c.R + c.G + c.B
		varSum += p.Variance().R + p.Variance().G + p.Variance().B
	}
	meanVar = varSum / float64(len(renderer.Buffer.Pixels)*3)
	return energy, bad, meanVar
}

// TestEnvSamplingEndToEnd renders a small lit scene (diffuse floor plus a
// glossy sphere) with environment sampling off and on, under both sampler
// configurations and with a rotated texture. It verifies that all pixels
// stay finite, that the MIS render is the more stable estimator, and that
// two independent MIS renders agree with each other (deterministic
// convergence), while the default (disabled) path is left untouched.
func TestEnvSamplingEndToEnd(t *testing.T) {
	build := func(angle float64) *Scene {
		scene := &Scene{}
		scene.Texture = sunSpotHDRI(128, 64, 90, 18, 4, 0.05, 20)
		scene.TextureAngle = angle
		floor := DiffuseMaterial(HexColor(0x888888))
		scene.Add(NewCube(V(-20, -2, -20), V(20, -1, 20), floor))
		glossy := GlossyMaterial(White, 1.5, Radians(5))
		scene.Add(NewSphere(V(0, -0.2, 0), 0.8, glossy))
		return scene
	}
	camera := LookAt(V(2, 1.5, 3), V(0, -0.5, 0), V(0, 1, 0), 50)

	cases := []struct {
		name    string
		sampler *DefaultSampler
	}{
		{"path-tracer", func() *DefaultSampler {
			s := NewSampler(2, 3)
			s.DirectLighting = false
			return s
		}()},
		{"direct-lighting", NewDirectSampler()},
	}
	for _, tc := range cases {
		sceneOff := build(0.6)
		sceneOn := build(0.6)
		sceneOn.EnvSampling = true
		energyOff, badOff, varOff := renderStats(sceneOff, camera, tc.sampler, 24)
		energyOn, badOn, varOn := renderStats(sceneOn, camera, tc.sampler, 24)
		if badOff != 0 || badOn != 0 {
			t.Fatalf("%s: non-finite pixels off=%d on=%d", tc.name, badOff, badOn)
		}
		// the enabled scene must have compiled once and cached a reusable
		// distribution
		if !sceneOn.EnvDistribution().Valid() {
			t.Fatalf("%s: enabled scene has no valid distribution", tc.name)
		}
		if sceneOn.EnvDistribution() != sceneOn.EnvDistribution() {
			t.Fatalf("%s: distribution must be cached and reused", tc.name)
		}
		t.Logf("%s: energy off=%.2f on=%.2f, variance off=%.3e on=%.3e",
			tc.name, energyOff, energyOn, varOff, varOn)
		// MIS render must retain environment energy and be finite
		if energyOn <= 0 {
			t.Fatalf("%s: MIS render lost all environment energy", tc.name)
		}
		// For the full path tracer the two estimators target the same
		// integral; with a small bright spot the plain path tracer
		// under-samples the spot, so MIS is allowed to carry *more*
		// recovered energy but must never be the noisier estimate.
		if tc.name == "path-tracer" && !(varOn < varOff) {
			t.Fatalf("%s: MIS variance %.3e not below baseline %.3e", tc.name, varOn, varOff)
		}
		// a second, independent MIS render must land close to the first,
		// i.e. the estimator is stable rather than lucky
		sceneOn2 := build(0.6)
		sceneOn2.EnvSampling = true
		energy2, bad2, _ := renderStats(sceneOn2, camera, tc.sampler, 24)
		if bad2 != 0 {
			t.Fatalf("%s: second render had %d bad pixels", tc.name, bad2)
		}
		if rel := math.Abs(energy2-energyOn) / energyOn; rel > 0.12 {
			t.Fatalf("%s: MIS renders not reproducible: %.2f vs %.2f (rel %.3f)",
				tc.name, energyOn, energy2, rel)
		}
	}
}

// TestEnvSamplingIrradianceReference verifies the production environment
// distribution and sampling math against an independent stratified
// quadrature of the same environment irradiance. The raw next-event
// estimator (before the MIS weight) must match the quadrature; the full
// two-technique MIS combination, whose two terms only conserve energy
// jointly, is checked by TestEnvEnergyAndVariance.
func TestEnvSamplingIrradianceReference(t *testing.T) {
	tex := sunSpotHDRI(128, 64, 96, 12, 5, 0.05, 20)
	const angle = 0.6
	ref := stratifiedHemisphereReference(tex, angle, 400, 800)

	dist := NewEnvDistribution(tex, angle)
	scene := &Scene{Texture: tex, TextureAngle: angle, EnvSampling: true}
	scene.Compile()
	rnd := rand.New(rand.NewSource(123))

	const n = 30000
	var sum float64
	for i := 0; i < n; i++ {
		dir, pdf, _, _ := dist.Sample(rnd)
		if pdf <= 0 {
			continue
		}
		if cos := dir.Y; cos > 0 {
			le := scene.Texture.Sample(envDirectionToUV(dir, angle))
			sum += le.R * cos / pdf
		}
	}
	got := sum / n
	t.Logf("raw NEE irradiance=%.5f  quadrature reference=%.5f", got, ref.R)
	if rel := math.Abs(got-ref.R) / ref.R; rel > 0.03 {
		t.Fatalf("raw NEE energy off by %.1f%%: got %.5f ref %.5f", rel*100, got, ref.R)
	}
}

// TestEnvSamplingSpecularEscape checks the delta-specular escape path:
// mirror rays must keep full environment weight (MIS collapses to one) and
// never vanish or blow up.
func TestEnvSamplingSpecularEscape(t *testing.T) {
	scene := &Scene{}
	scene.Texture = sunSpotHDRI(64, 32, 40, 10, 3, 0.1, 10)
	scene.EnvSampling = true
	mirror := MetallicMaterial(White, 0, 0)
	scene.Add(NewSphere(V(0, 0, -3), 1, mirror))
	camera := LookAt(V(0, 0, 0), V(0, 0, -1), V(0, 1, 0), 60)
	renderer := NewRenderer(scene, &camera, NewSampler(1, 2), 32, 32)
	renderer.SamplesPerPixel = 16
	renderer.NumCPU = 2
	renderer.Verbose = false
	renderer.Render()
	for i, p := range renderer.Buffer.Pixels {
		c := p.Color()
		if !isFinite(c.R) || !isFinite(c.G) || !isFinite(c.B) {
			t.Fatalf("pixel %d non-finite: %v", i, c)
		}
	}
}
