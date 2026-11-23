package pt

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Test fixtures: two-layer density peaks inside a grid volume, point light.
// ---------------------------------------------------------------------------

const testN = 40

// twoPeakGrid builds a W*H*D grid whose density along x is the sum of two
// narrow Gaussian peaks. y and z are constant (slab geometry), which makes
// the 1D numerical reference along a ray exact.
func twoPeakGrid(w, h, d int) []float64 {
	data := make([]float64, w*h*d)
	c1, c2 := -0.45, 0.45
	width := 0.05
	amp1, amp2 := 1.0, 0.6
	for z := 0; z < d; z++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				// map voxel center to world x in [-1, 1]
				wx := -1 + 2*(float64(x)+0.5)/float64(w)
				density := amp1*math.Exp(-(wx-c1)*(wx-c1)/(2*width*width)) +
					amp2*math.Exp(-(wx-c2)*(wx-c2)/(2*width*width))
				// add a tiny negative component to exercise clamping
				if x == 0 {
					density = -0.25
				}
				data[x+y*w+z*w*h] = density
			}
		}
	}
	return data
}

func testMedium() *Medium {
	return &Medium{
		Step:             0.02,
		Absorption:       Color{0.2, 0.15, 0.1},
		Scattering:       Color{0.8, 0.7, 0.6},
		Phase:            IsotropicPhase{},
		MaxScatterEvents: 4,
	}
}

func testVolume() *Volume {
	box := Box{Vector{-1, -1, -1}, Vector{1, 1, 1}}
	v := NewVolumeData(box, testN, testN, testN, twoPeakGrid(testN, testN, testN), testMedium())
	v.Compile()
	return v
}

// referenceDensity evaluates the analytic two-peak field used to build the
// grid, so the reference integrator is independent of the grid sampler.
func referenceDensity(x float64) float64 {
	c1, c2 := -0.45, 0.45
	width := 0.05
	amp1, amp2 := 1.0, 0.6
	d := amp1*math.Exp(-(x-c1)*(x-c1)/(2*width*width)) +
		amp2*math.Exp(-(x-c2)*(x-c2)/(2*width*width))
	if d < 0 {
		return 0
	}
	return d
}

// referenceTransmittance integrates optical depth along the x-axis ray with
// a fine uniform trapezoidal rule independent of the volume marcher.
func referenceTransmittance(sigmaT Color, x0, x1 float64) Color {
	const n = 200000
	dt := (x1 - x0) / n
	tau := [3]float64{}
	for i := 0; i <= n; i++ {
		x := x0 + float64(i)*dt
		d := referenceDensity(x)
		w := 1.0
		if i == 0 || i == n {
			w = 0.5
		}
		tau[0] += w * sigmaT.R * d * dt
		tau[1] += w * sigmaT.G * d * dt
		tau[2] += w * sigmaT.B * d * dt
	}
	return Color{math.Exp(-tau[0]), math.Exp(-tau[1]), math.Exp(-tau[2])}
}

func colorNear(a, b Color, tol float64) bool {
	return math.Abs(a.R-b.R) <= tol && math.Abs(a.G-b.G) <= tol &&
		math.Abs(a.B-b.B) <= tol
}

// ---------------------------------------------------------------------------
// 1. Transmittance against the independent numerical reference.
// ---------------------------------------------------------------------------

func TestVolumeTransmittanceReference(t *testing.T) {
	v := testVolume()
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
	got := v.Transmittance(ray, 0, 2)
	want := referenceTransmittance(v.medium.extinction(), -1, 1)
	if !colorNear(got, want, 0.02) {
		t.Fatalf("transmittance mismatch: got %v want %v", got, want)
	}
	// Tighter step must converge toward the reference: error should shrink.
	fine := *testMedium()
	fine.Step = 0.002
	fv := NewVolumeData(v.Box, testN, testN, testN,
		twoPeakGrid(testN, testN, testN), &fine)
	fv.Compile()
	gotFine := fv.Transmittance(ray, 0, 2)
	if !colorNear(gotFine, want, 0.002) {
		t.Fatalf("fine-step transmittance mismatch: got %v want %v", gotFine, want)
	}
}

func TestTransmittanceBoundsAndMonotonicity(t *testing.T) {
	v := testVolume()
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
	// Transmittance must stay within [exp(-tauMax), 1] and be monotone.
	prev := 1.0
	for i := 1; i <= 20; i++ {
		tEnd := 2 * float64(i) / 20
		tr := v.Transmittance(ray, 0, tEnd)
		if tr.R < math.Exp(-maxVolumeOpticalDepth)-1e-12 || tr.R > 1+1e-12 {
			t.Fatalf("transmittance out of bounds: %v", tr)
		}
		if tr.R > prev+1e-12 {
			t.Fatalf("transmittance not monotone at %v: %v > %v", tEnd, tr.R, prev)
		}
		prev = tr.R
	}
}

// ---------------------------------------------------------------------------
// 2. Adaptive stepping must not step over narrow peaks.
// ---------------------------------------------------------------------------

func TestAdaptiveStepNarrowPeak(t *testing.T) {
	// A single dense voxel surrounded by vacuum is the narrowest possible
	// peak. A naive marcher with a large step would miss it entirely.
	w := 8
	data := make([]float64, w*w*w)
	data[((w/2)+w*(w/2))+w*w*(w/2)] = 5.0 // single dense voxel at center
	m := &Medium{Step: 1.0, Absorption: Color{1, 1, 1}, Scattering: Black,
		Phase: IsotropicPhase{}, MaxScatterEvents: 0}
	v := NewVolumeData(Box{Vector{-1, -1, -1}, Vector{1, 1, 1}}, w, w, w, data, m)
	v.Compile()
	// Ray through the voxel center.
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
	tr := v.Transmittance(ray, 0, 2)
	// Voxel spans 1/8 of the ray; density 5, sigma=1 -> optical depth ~0.625.
	// Any result above ~0.95 means the peak was skipped.
	if tr.R > 0.95 {
		t.Fatalf("narrow single-voxel peak was skipped, transmittance=%v", tr)
	}
	// And it must agree with an exact box integral over the voxel extent.
	vx := 2.0 / float64(w)
	want := math.Exp(-5.0 * vx)
	if math.Abs(tr.R-want) > 0.02 {
		t.Fatalf("narrow peak transmittance %v, want ~%v", tr.R, want)
	}
}

// ---------------------------------------------------------------------------
// 3. Edge cases: empty grid, negative density, zero step, out of bounds,
//    long intervals, non-finite coefficients.
// ---------------------------------------------------------------------------

func TestEmptyVolume(t *testing.T) {
	v := NewVolumeData(Box{Vector{-1, -1, -1}, Vector{1, 1, 1}}, 0, 0, 0, nil, testMedium())
	v.Compile()
	ray := Ray{Vector{-2, 0, 0}, Vector{1, 0, 0}}
	if got := v.Transmittance(ray, 0, 4); !colorNear(got, White, 1e-12) {
		t.Fatalf("empty volume transmittance = %v, want White", got)
	}
	if got := v.Density(Vector{0, 0, 0}); got != 0 {
		t.Fatalf("empty volume density = %v, want 0", got)
	}
}

func TestNegativeAndOutOfBoundsDensity(t *testing.T) {
	v := testVolume()
	// Out-of-bounds points read as vacuum.
	for _, p := range []Vector{{-5, 0, 0}, {5, 0, 0}, {0, 10, 0}, {0, 0, -10}} {
		if got := v.Density(p); got != 0 {
			t.Fatalf("out-of-bounds density at %v = %v, want 0", p, got)
		}
	}
	// Non-finite coordinates return zero instead of NaN.
	for _, p := range []Vector{{math.NaN(), 0, 0}, {0, math.Inf(1), 0}} {
		if got := v.Density(p); got != 0 {
			t.Fatalf("non-finite density at %v = %v, want 0", p, got)
		}
	}
	// The negative voxel in the fixture must clamp to zero.
	p := rayPositionForVoxel(v, 0, testN/2, testN/2)
	if got := v.Density(p); got != 0 {
		t.Fatalf("negative density not clamped: %v", got)
	}
}

func rayPositionForVoxel(v *Volume, i, j, k int) Vector {
	// Inverse of the world->grid mapping for voxel center i along x.
	wx := -1 + 2*(float64(i)+0.5)/float64(v.W)
	wy := -1 + 2*(float64(j)+0.5)/float64(v.H)
	wz := -1 + 2*(float64(k)+0.5)/float64(v.D)
	return Vector{wx, wy, wz}
}

func TestZeroAndNonFiniteStep(t *testing.T) {
	for _, step := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		m := testMedium()
		m.Step = step
		v := NewVolumeData(Box{Vector{-1, -1, -1}, Vector{1, 1, 1}},
			testN, testN, testN, twoPeakGrid(testN, testN, testN), m)
		v.Compile()
		if v.medium.Step <= 0 || !isFinite(v.medium.Step) {
			t.Fatalf("step %v not normalized: %v", step, v.medium.Step)
		}
		ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
		tr := v.Transmittance(ray, 0, 2)
		if !isFinite(tr.R) || tr.R < 0 || tr.R > 1 {
			t.Fatalf("step %v produced invalid transmittance %v", step, tr)
		}
	}
}

func TestNonFiniteCoefficients(t *testing.T) {
	m := testMedium()
	m.Absorption = Color{math.NaN(), math.Inf(1), math.Inf(-1)}
	m.Scattering = Color{-5, 1e30, 0}
	m.G = math.NaN()
	m.MaxScatterEvents = 1 << 30
	v := NewVolumeData(Box{Vector{-1, -1, -1}, Vector{1, 1, 1}},
		testN, testN, testN, twoPeakGrid(testN, testN, testN), m)
	v.Compile()
	cfg := v.medium
	for _, c := range []float64{cfg.Absorption.R, cfg.Absorption.G, cfg.Absorption.B,
		cfg.Scattering.R, cfg.Scattering.G, cfg.Scattering.B} {
		if !isFinite(c) || c < 0 || c > maxVolumeCoefficient {
			t.Fatalf("coefficient not sanitized: %v", c)
		}
	}
	if cfg.MaxScatterEvents > maxScatterEventsLimit {
		t.Fatalf("scatter event limit not capped: %v", cfg.MaxScatterEvents)
	}
	if cfg.Phase == nil {
		t.Fatal("nil/NaN G did not produce a default phase function")
	}
}

func TestExtremelyLongInterval(t *testing.T) {
	v := testVolume()
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
	// An astronomically long interval must stay finite and bounded below.
	tr := v.Transmittance(ray, 0, 1e15)
	lo := math.Exp(-maxVolumeOpticalDepth)
	for _, c := range []float64{tr.R, tr.G, tr.B} {
		if !isFinite(c) || c < lo-1e-12 || c > 1+1e-12 {
			t.Fatalf("long-interval transmittance invalid: %v", tr)
		}
	}
	// The marcher must terminate (segment cap) rather than hang.
	done := make(chan struct{})
	go func() {
		v.Transmittance(ray, 0, 1e15)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("marcher did not terminate on an extremely long interval")
	}
}

// ---------------------------------------------------------------------------
// 4. Legacy isosurface behavior is preserved when no Medium is configured.
// ---------------------------------------------------------------------------

func TestLegacyIsosurfaceUnchanged(t *testing.T) {
	// A volume with windows but no medium must still intersect its isosurface
	// and report no media to the scene.
	mat := GlossyMaterial(White, 1.3, 0)
	windows := []VolumeWindow{{0.3, 0.7, mat}}
	data := make([]float64, testN*testN*testN)
	// Fill the central region with density inside the window.
	for z := 0; z < testN; z++ {
		for y := 0; y < testN; y++ {
			for x := 0; x < testN; x++ {
				if x >= testN/4 && x < 3*testN/4 {
					data[x+y*testN+z*testN*testN] = 0.5
				}
			}
		}
	}
	v := NewVolumeData(Box{Vector{-1, -1, -1}, Vector{1, 1, 1}},
		testN, testN, testN, data, nil)
	v.Windows = windows
	v.Compile()
	if v.IsMedium() {
		t.Fatal("volume without Medium should not be a medium")
	}
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
	hit := v.Intersect(ray)
	if !hit.Ok() {
		t.Fatal("legacy isosurface intersection lost")
	}
	if hit.Shape != v {
		t.Fatal("legacy hit shape mismatch")
	}
	scene := Scene{}
	scene.Add(v)
	scene.Compile()
	if scene.HasMedia() {
		t.Fatal("isosurface volume registered as medium")
	}
	if got := scene.Transmittance(ray, 0, 2); !colorNear(got, White, 1e-12) {
		t.Fatalf("isosurface scene transmittance = %v, want White", got)
	}
}

// ---------------------------------------------------------------------------
// 5. Shadow, reflection and refraction rays share the same density field.
// ---------------------------------------------------------------------------

func TestAllRayTypesUseSameDensityField(t *testing.T) {
	v := testVolume()
	// Transmittance is a pure function of the ray segment; the same segment
	// expressed as a shadow, reflection or refraction ray must be identical.
	origin := Vector{-1, 0.3, -0.2}
	dir := Vector{1, 0, 0}
	ray := Ray{origin, dir}
	base := v.Transmittance(ray, 0.1, 1.5)
	// Reverse direction (reflection-like): transmittance over an interval is
	// independent of traversal orientation.
	rev := Ray{origin.Add(dir.MulScalar(2)), dir.Negate()}
	revTr := v.Transmittance(rev, 0.5, 1.9)
	if !colorNear(base, revTr, 1e-9) {
		t.Fatalf("reverse ray transmittance differs: %v vs %v", base, revTr)
	}
	// Scene-level entry point used by shadow/reflection/refraction rays.
	scene := Scene{}
	scene.Add(v)
	scene.Compile()
	sceneTr := scene.Transmittance(ray, 0.1, 1.5)
	if !colorNear(base, sceneTr, 1e-12) {
		t.Fatalf("scene vs volume transmittance differ: %v vs %v", base, sceneTr)
	}
}

// ---------------------------------------------------------------------------
// 6. Scattering order and convergence against the numerical reference.
// ---------------------------------------------------------------------------

// uncollidedBaseline reproduces the order-0 radiance independently of
// deltaTrack. In this fixture the medium is not a surface, so the ray
// misses every shape except the light sphere. A direct hit on an emissive
// surface returns its full emittance (the legacy surface convention); the
// media path dims exactly that by beam transmittance up to the hit point.
func uncollidedBaseline(scene *Scene, s *DefaultSampler, ray Ray) Color {
	hit := scene.Intersect(ray)
	if !hit.Ok() {
		return scene.Transmittance(ray, 0, INF).Mul(s.sampleEnvironment(scene, ray))
	}
	tr := scene.Transmittance(ray, 0, hit.T)
	info := hit.Info(ray)
	return tr.Mul(info.Material.Color.MulScalar(info.Material.Emittance))
}

func TestSingleScatterReference(t *testing.T) {
	v := testVolume()
	// Point light on the axis beyond the second peak.
	light := NewSphere(Vector{1.5, 0, 0}, 0.05, LightMaterial(White, 50))
	scene := Scene{Color: Black}
	scene.Add(v)
	scene.Add(light)
	scene.Compile()
	s := NewDirectSampler()
	s.LightMode = LightModeAll
	rnd := rand.New(rand.NewSource(42))
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
	got := s.SingleScatter(&scene, ray, rnd)
	want := referenceSingleScatter(v.medium, ray, light.(*Sphere))
	rel := math.Abs(got.R-want.R) / math.Max(want.R, 1e-9)
	if rel > 0.05 {
		t.Fatalf("single scatter: got %v want %v (rel err %v)", got, want, rel)
	}
	if got.R <= 0 {
		t.Fatal("single scatter returned no energy")
	}
}

// referenceSingleScatter is an independent fine-grid evaluation of the
// single-scattering integral for the isotropic slab / point-light setup.
// Cumulative optical depth is precomputed once, giving O(n) rather than
// O(n^2) transmittance lookups.
func referenceSingleScatter(m *Medium, ray Ray, light *Sphere) Color {
	const n = 4000
	x0, x1 := -1.0, 1.0
	dt := (x1 - x0) / n
	sigmaT := m.extinction()
	sigmaS := m.Scattering
	phase := 1.0 / (4 * math.Pi)
	// Cumulative optical depth at each grid node (trapezoidal).
	tau := make([][3]float64, n+1)
	for i := 1; i <= n; i++ {
		xl := x0 + float64(i-1)*dt
		xr := x0 + float64(i)*dt
		dl, dr := referenceDensity(xl), referenceDensity(xr)
		dAvg := (dl + dr) / 2
		for c := 0; c < 3; c++ {
			st := []float64{sigmaT.R, sigmaT.G, sigmaT.B}[c]
			tau[i][c] = tau[i-1][c] + st*dAvg*dt
		}
	}
	tauTotal := tau[n]
	trAt := func(i int) [3]float64 {
		var out [3]float64
		for c := 0; c < 3; c++ {
			out[c] = math.Exp(-tau[i][c])
		}
		return out
	}
	lc := light.Center
	lr := light.Radius
	le := light.Material.Emittance
	integral := [3]float64{}
	for i := 0; i <= n; i++ {
		x := x0 + float64(i)*dt
		d := referenceDensity(x)
		if d <= 0 {
			continue
		}
		trTo := trAt(i)
		// Point-light NEE: projected solid angle pi*r^2/dist^2.
		dist := lc.X - x
		geom := math.Pi * lr * lr / (dist * dist)
		// Transmittance from x to the light center via the cumulative total.
		var trLight [3]float64
		for c := 0; c < 3; c++ {
			trLight[c] = math.Exp(-(tauTotal[c] - tau[i][c]))
		}
		w := 1.0
		if i == 0 || i == n {
			w = 0.5
		}
		for c := 0; c < 3; c++ {
			sigS := []float64{sigmaS.R, sigmaS.G, sigmaS.B}[c]
			integral[c] += w * sigS * d * dt * phase * le * geom *
				trTo[c] * trLight[c]
		}
	}
	return Color{integral[0], integral[1], integral[2]}
}

func TestScatteringOrder(t *testing.T) {
	v := testVolume()
	light := NewSphere(Vector{1.5, 0, 0}, 0.05, LightMaterial(White, 50))
	scene := Scene{Color: Black}
	scene.Add(v)
	scene.Add(light)
	scene.Compile()
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}

	// Order-0 is deterministic: with ScatterEvents=0 the stochastic term is
	// disabled, so repeated renders must return bit-identical results.
	s0 := NewDirectSampler()
	s0.ScatterEvents = 0
	s0.LightMode = LightModeAll
	a := s0.Sample(&scene, ray, rand.New(rand.NewSource(7)))
	b := s0.Sample(&scene, ray, rand.New(rand.NewSource(99)))
	if !colorNear(a, b, 0) {
		t.Fatalf("order-0 render is not deterministic: %v vs %v", a, b)
	}
	for _, c := range []float64{a.R, a.G, a.B} {
		if !isFinite(c) || c < 0 {
			t.Fatalf("order-0 result invalid: %v", a)
		}
	}
	// Order-0 must equal the uncollided (surface-path) baseline computed with
	// the stochastic walk disabled: transmittance dims the direct light, no
	// in-scattering is added. Establish that baseline independently.
	baseline := uncollidedBaseline(&scene, s0, ray)
	if !colorNear(a, baseline, 1e-9) {
		t.Fatalf("order-0 = %v, want uncollided baseline %v", a, baseline)
	}

	// Adding scattering orders can only add non-negative energy. Use an
	// absorption-free medium so every added order contributes positively in
	// expectation; compare ensemble means over independent random seeds.
	pure := testMedium()
	pure.Absorption = Black
	pv := NewVolumeData(v.Box, testN, testN, testN,
		twoPeakGrid(testN, testN, testN), pure)
	pv.Compile()
	pscene := Scene{Color: Black}
	pscene.Add(pv)
	pscene.Add(light)
	pscene.Compile()
	meanR := func(events, n int) float64 {
		s := NewDirectSampler()
		s.ScatterEvents = events
		s.LightMode = LightModeAll
		var sum float64
		for i := 0; i < n; i++ {
			sum += s.Sample(&pscene, ray, rand.New(rand.NewSource(int64(500+i)))).R
		}
		return sum / float64(n)
	}
	m0 := meanR(0, 64)
	m1 := meanR(1, 64)
	m2 := meanR(3, 64)
	if !(m1 >= m0-1e-9) {
		t.Fatalf("first scattering order reduced energy: %v < %v", m1, m0)
	}
	if !(m2 >= m1-1e-6) {
		t.Fatalf("higher scattering orders reduced energy: %v < %v", m2, m1)
	}
}

func TestStochasticConvergenceAndNoise(t *testing.T) {
	v := testVolume()
	light := NewSphere(Vector{1.5, 0, 0}, 0.05, LightMaterial(White, 50))
	scene := Scene{Color: Black}
	scene.Add(v)
	scene.Add(light)
	scene.Compile()
	ray := Ray{Vector{-1, 0, 0}, Vector{1, 0, 0}}
	s := NewDirectSampler()
	s.ScatterEvents = 2
	s.LightMode = LightModeAll
	// Independent samples: the mean must converge (standard error ~ 1/sqrt(N)).
	mean := func(n int) (float64, float64) {
		var sum, sum2 float64
		for i := 0; i < n; i++ {
			rnd := rand.New(rand.NewSource(int64(1000 + i)))
			c := s.Sample(&scene, ray, rnd).R
			sum += c
			sum2 += c * c
		}
		m := sum / float64(n)
		variance := sum2/float64(n) - m*m
		return m, math.Sqrt(math.Max(0, variance) / float64(n))
	}
	m16, se16 := mean(16)
	m256, se256 := mean(256)
	if !isFinite(m256) || m256 < 0 {
		t.Fatalf("stochastic mean invalid: %v", m256)
	}
	// Standard error must shrink roughly with 1/sqrt(N): 16x samples -> ~4x.
	if se256 > se16 {
		t.Fatalf("noise did not reduce with more samples: se16=%v se256=%v", se16, se256)
	}
	// The two means should be statistically compatible.
	if math.Abs(m16-m256) > 4*math.Max(se16, se256) {
		t.Fatalf("means diverge beyond 4 SE: %v vs %v", m16, m256)
	}
}

// ---------------------------------------------------------------------------
// 7. Phase functions.
// ---------------------------------------------------------------------------

func TestIsotropicPhase(t *testing.T) {
	p := IsotropicPhase{}
	if got := p.Evaluate(Vector{1, 0, 0}, Vector{0, 1, 0}); math.Abs(got-1/(4*math.Pi)) > 1e-12 {
		t.Fatalf("isotropic evaluate = %v", got)
	}
	rnd := rand.New(rand.NewSource(1))
	for i := 0; i < 100; i++ {
		out, pdf := p.Sample(Vector{1, 0, 0}, rnd)
		if math.Abs(out.Length()-1) > 1e-9 {
			t.Fatalf("phase sample not unit length: %v", out)
		}
		if math.Abs(pdf-1/(4*math.Pi)) > 1e-12 {
			t.Fatalf("isotropic pdf = %v", pdf)
		}
	}
}

func TestHenyeyGreensteinNormalization(t *testing.T) {
	// The HG phase integrates to one over the sphere; verify numerically.
	for _, g := range []float64{-0.5, 0.0, 0.7} {
		p := HenyeyGreensteinPhase{g}
		const n = 2000
		sum := 0.0
		for i := 0; i <= n; i++ {
			cos := -1 + 2*float64(i)/float64(n)
			w := 1.0
			if i == 0 || i == n {
				w = 0.5
			}
			sum += w * p.Evaluate(Vector{1, 0, 0}, Vector{cos,
				math.Sqrt(math.Max(0, 1-cos*cos)), 0}) * (2 * math.Pi) * (2.0 / n)
		}
		if math.Abs(sum-1) > 0.02 {
			t.Fatalf("HG(g=%v) integrates to %v, want 1", g, sum)
		}
	}
}

func TestHenyeyGreensteinSampleDirection(t *testing.T) {
	p := HenyeyGreensteinPhase{0.8} // strongly forward scattering
	rnd := rand.New(rand.NewSource(3))
	in := Vector{1, 0, 0}
	var cosSum float64
	const n = 2000
	for i := 0; i < n; i++ {
		out, pdf := p.Sample(in, rnd)
		if math.Abs(out.Length()-1) > 1e-9 || !isFinite(pdf) || pdf <= 0 {
			t.Fatalf("invalid HG sample: %v pdf %v", out, pdf)
		}
		cosSum += out.Dot(in)
	}
	if cosSum/float64(n) < 0.5 {
		t.Fatalf("forward HG phase produced mean cosine %v, want > 0.5", cosSum/float64(n))
	}
}
