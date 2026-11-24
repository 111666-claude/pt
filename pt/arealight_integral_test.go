package pt

import (
	"math"
	"math/rand"
	"testing"
)

// polygonIrradiance computes the analytic irradiance at p from a flat,
// unoccluded polygon with constant two-sided radiance L, following the
// edge-vector solid-angle formulation. All polygon vertices must lie in one
// plane; the plane normal is derived from the first three vertices.
//
// E = (L / 2) * sum_i gamma_i * (r_i x r_{i+1}) . n / |r_i x r_{i+1}|
//
// where gamma_i is the angle between consecutive vertex directions. The 1/2
// factor is required: for an infinite plane the sum converges to 2*pi and
// the correct Lambertian limit is E = L*pi.
func polygonIrradiance(p, n Vector, L float64, verts []Vector) float64 {
	var sum float64
	m := len(verts)
	for i := 0; i < m; i++ {
		a := verts[i].Sub(p)
		b := verts[(i+1)%m].Sub(p)
		cross := a.Cross(b)
		sinGamma := cross.Length() / (a.Length() * b.Length())
		cosGamma := a.Dot(b) / (a.Length() * b.Length())
		gamma := math.Atan2(sinGamma, cosGamma)
		sum += gamma * cross.Dot(n) / cross.Length()
	}
	return L / 2 * math.Abs(sum)
}

// meshOwnsHit reports whether a shadow ray hit landed on a triangle that
// belongs to the given emissive mesh.
func meshOwnsHit(mesh *Mesh, hit Hit) bool {
	tri, ok := hit.Shape.(*Triangle)
	if !ok {
		return false
	}
	for _, candidate := range mesh.Triangles {
		if candidate == tri {
			return true
		}
	}
	return false
}

// quadMesh builds a two-triangle emissive quad in the plane y=height,
// spanning [x0,x1] x [z0,z1], facing -Y.
func quadMesh(x0, x1, z0, z1, height, emittance float64) *Mesh {
	v000 := V(x0, height, z0)
	v100 := V(x1, height, z0)
	v110 := V(x1, height, z1)
	v010 := V(x0, height, z1)
	m := LightMaterial(White, emittance)
	t1 := NewTriangle(v000, v100, v110, Vector{}, Vector{}, Vector{}, m)
	t2 := NewTriangle(v000, v110, v010, Vector{}, Vector{}, Vector{}, m)
	return NewMesh([]*Triangle{t1, t2})
}

// estimateGroundIrradiance renders the direct-only irradiance at a single
// point on an upward-facing diffuse ground plane. Returns the mean and
// sample variance of the per-sample estimates.
func estimateGroundIrradiance(t *testing.T, scene *Scene, p Vector, samples int, rnd *rand.Rand, soft bool) (mean, variance float64) {
	scene.Compile()
	sampler := NewDirectSampler()
	sampler.SoftShadows = soft
	n := Ray{Origin: p, Direction: V(0, 1, 0)}
	var sum, sumSq float64
	for i := 0; i < samples; i++ {
		c := sampler.sampleLights(scene, n, rnd)
		v := c.G // white light: all channels equal
		sum += v
		sumSq += v * v
	}
	mean = sum / float64(samples)
	variance = sumSq/float64(samples) - mean*mean
	return
}

func TestSingleTriangleLightMatchesAnalytic(t *testing.T) {
	// single triangle light above a ground point at the origin
	tri := emitTri(V(-1, 2, -1), V(1, 2, -1), V(0, 2, 1), 8)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(tri)
	scene.Add(ground)

	p := V(0, 0, 0)
	analytic := polygonIrradiance(p, V(0, 1, 0), 8,
		[]Vector{tri.V1, tri.V2, tri.V3})

	rnd := newTestRNG()
	mean, variance := estimateGroundIrradiance(t, &scene, p, 40000, rnd, true)
	if math.Abs(mean-analytic)/analytic > 0.03 {
		t.Fatalf("triangle light irradiance = %v, analytic = %v (rel err %.3f)",
			mean, analytic, math.Abs(mean-analytic)/analytic)
	}
	if variance <= 0 {
		t.Fatalf("variance should be positive for a soft light")
	}
	t.Logf("triangle: mc=%.4f analytic=%.4f relerr=%.4f var=%.4f",
		mean, analytic, math.Abs(mean-analytic)/analytic, variance)
}

func TestMultiTriangleMeshLightMatchesAnalytic(t *testing.T) {
	// quad light built from two triangles, displaced so the origin is off-center
	mesh := quadMesh(-1.5, 1.0, -0.8, 1.2, 2.5, 6)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(mesh)
	scene.Add(ground)
	if len(scene.Lights) != 1 {
		t.Fatalf("mesh should be registered as one light, got %d", len(scene.Lights))
	}

	p := V(0, 0, 0)
	verts := []Vector{V(-1.5, 2.5, -0.8), V(1.0, 2.5, -0.8), V(1.0, 2.5, 1.2), V(-1.5, 2.5, 1.2)}
	analytic := polygonIrradiance(p, V(0, 1, 0), 6, verts)

	rnd := newTestRNG()
	mean, variance := estimateGroundIrradiance(t, &scene, p, 60000, rnd, true)
	if math.Abs(mean-analytic)/analytic > 0.03 {
		t.Fatalf("mesh light irradiance = %v, analytic = %v (rel err %.3f)",
			mean, analytic, math.Abs(mean-analytic)/analytic)
	}
	t.Logf("mesh quad: mc=%.4f analytic=%.4f relerr=%.4f var=%.4f",
		mean, analytic, math.Abs(mean-analytic)/analytic, variance)
}

func TestTwoSeparateEmissiveMeshesNoDoubleCount(t *testing.T) {
	l1 := quadMesh(-2, -0.2, -1, 1, 2, 5)
	l2 := quadMesh(0.2, 2, -1, 1, 2, 5)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(l1)
	scene.Add(l2)
	scene.Add(ground)
	if len(scene.Lights) != 2 {
		t.Fatalf("expected 2 lights, got %d", len(scene.Lights))
	}
	p := V(0, 0, 0)
	v1 := polygonIrradiance(p, V(0, 1, 0), 5,
		[]Vector{V(-2, 2, -1), V(-0.2, 2, -1), V(-0.2, 2, 1), V(-2, 2, 1)})
	v2 := polygonIrradiance(p, V(0, 1, 0), 5,
		[]Vector{V(0.2, 2, -1), V(2, 2, -1), V(2, 2, 1), V(0.2, 2, 1)})
	analytic := v1 + v2

	rnd := newTestRNG()
	// LightModeAll: each light sampled once per estimate — direct addition
	mean, _ := estimateGroundIrradiance(t, &scene, p, 40000, rnd, true)
	if math.Abs(mean-analytic)/analytic > 0.03 {
		t.Fatalf("two-mesh irradiance = %v, analytic = %v", mean, analytic)
	}

	// LightModeRandom: one light picked uniformly and weighted by nLights —
	// same expected value, proving no double counting in either mode
	rnd2 := rand.New(rand.NewSource(7))
	scene2 := Scene{}
	scene2.Add(l1)
	scene2.Add(l2)
	scene2.Add(ground)
	scene2.Compile()
	sampler := NewDirectSampler()
	sampler.LightMode = LightModeRandom
	sampler.SoftShadows = true
	n := Ray{Origin: p, Direction: V(0, 1, 0)}
	var sum, sumSq float64
	const k = 100000
	for i := 0; i < k; i++ {
		v := sampler.sampleLights(&scene2, n, rnd2).G
		sum += v
		sumSq += v * v
	}
	meanRandom := sum / k
	variance := sumSq/k - meanRandom*meanRandom
	stdErr := math.Sqrt(variance / k)
	// statistically correct MC check: the analytic value must lie within
	// four standard errors of the Monte Carlo mean
	if math.Abs(meanRandom-analytic) > 4*stdErr {
		t.Fatalf("random-light mode irradiance = %v, analytic = %v (4*SE = %v)",
			meanRandom, analytic, 4*stdErr)
	}
	t.Logf("two lights: all=%.4f random=%.4f analytic=%.4f", mean, meanRandom, analytic)
}

func TestFullyOccludedSceneIsBlack(t *testing.T) {
	mesh := quadMesh(-1, 1, -1, 1, 3, 10)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	// opaque slab between the light and the ground point
	occluder := NewCube(V(-10, 1.4, -10), V(10, 1.6, 10), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(mesh)
	scene.Add(occluder)
	scene.Add(ground)
	rnd := newTestRNG()
	mean, _ := estimateGroundIrradiance(t, &scene, V(0, 0, 0), 5000, rnd, true)
	if mean != 0 {
		t.Fatalf("fully occluded irradiance = %v, want 0", mean)
	}
}

func TestPartialOcclusionShadowEdge(t *testing.T) {
	// half the light is covered by an occluder wall at x=0. The wall rises
	// above the light (y=3) so the covered half is fully black while the
	// open half keeps a smooth penumbra toward x=0.
	mesh := quadMesh(-2, 2, -1, 1, 3, 10)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	occluder := NewCube(V(-10, 0.05, -10), V(0, 11, 10), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(mesh)
	scene.Add(occluder)
	scene.Add(ground)

	rnd := newTestRNG()
	// point well under the open half receives light
	lit, _ := estimateGroundIrradiance(t, &scene, V(1.5, 0, 0), 8000, rnd, true)
	// point well under the occluded half is black
	dark, _ := estimateGroundIrradiance(t, &scene, V(-1.5, 0, 0), 8000, rand.New(rand.NewSource(99)), true)
	if lit <= 0 {
		t.Fatalf("lit side must receive light, got %v", lit)
	}
	if dark != 0 {
		t.Fatalf("occluded side must be black, got %v", dark)
	}
	// shadow edge: a point just inside the open half is dimmer than one
	// near the center of the open half
	edge, _ := estimateGroundIrradiance(t, &scene, V(0.05, 0, 0), 8000, rand.New(rand.NewSource(123)), true)
	if edge >= lit {
		t.Fatalf("shadow edge (%v) should be dimmer than fully lit point (%v)", edge, lit)
	}
	t.Logf("shadow profile: lit=%.4f edge=%.4f dark=%.4f", lit, edge, dark)
}

func TestBackSideEmissionIsTwoSided(t *testing.T) {
	// light faces -Y (winding order); ground is below it, so it sees the
	// front. Flip the winding and verify the back side still emits, since
	// lights in this engine are two-sided emitters.
	tri1 := emitTri(V(-1, 2, -1), V(1, 2, -1), V(0, 2, 1), 8)
	tri2 := emitTri(V(-1, 2, -1), V(0, 2, 1), V(1, 2, -1), 8) // reversed winding
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	s1, s2 := Scene{}, Scene{}
	s1.Add(tri1)
	s1.Add(ground)
	s2.Add(tri2)
	s2.Add(ground)
	rnd := newTestRNG()
	front, _ := estimateGroundIrradiance(t, &s1, V(0, 0, 0), 20000, rnd, true)
	back, _ := estimateGroundIrradiance(t, &s2, V(0, 0, 0), 20000, rand.New(rand.NewSource(55)), true)
	if math.Abs(front-back)/front > 0.04 {
		t.Fatalf("two-sided emission broken: front=%v back=%v", front, back)
	}
	t.Logf("two-sided: front=%.4f back=%.4f", front, back)
}

func TestAreaSamplingBeatsBoundingDiskVariance(t *testing.T) {
	// large quad light: the old bounding-disk approximation wastes a large
	// fraction of samples outside the light surface and compensates with the
	// coverage heuristic, producing high variance. Area sampling should have
	// strictly lower variance at the same sample count. The geometry is kept
	// in the regime where the legacy solid-angle formula is finite
	// (radius < distance), since outside it the legacy code degenerates.
	mesh := quadMesh(-1.5, 1.5, -1.5, 1.5, 4, 4)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(mesh)
	scene.Add(ground)
	scene.Compile()
	p := V(0, 0, 0)

	// new area estimator
	rnd := newTestRNG()
	_, areaVar := estimateGroundIrradiance(t, &scene, p, 20000, rnd, true)

	// legacy bounding-disk estimator, reproduced inline. Every disk sample
	// that does not land on the emissive surface contributes zero (this is
	// the source of the bright-spot / light-leak noise: most disk samples
	// miss a thin or large-area light). Variance is accumulated over ALL
	// samples, including the zeros, with Welford's algorithm for stability.
	box := mesh.BoundingBox()
	radius := box.OuterRadius()
	center := box.Center()
	rnd2 := rand.New(rand.NewSource(42))
	n := Ray{Origin: p, Direction: V(0, 1, 0)}
	hyp := center.Sub(n.Origin).Length()
	// clamp the asin argument: for a light wider than its distance, the
	// legacy geometry goes singular and the coverage saturates at 1
	sinArg := math.Min(radius/hyp, 1)
	theta := math.Asin(sinArg)
	adj := radius / math.Tan(theta)
	d := math.Cos(theta) * adj
	r := math.Sin(theta) * adj
	coverage := math.Min((r*r)/(d*d), 1)

	const k = 20000
	var mean, m2 float64
	accepted := 0
	for i := 0; i < k; i++ {
		var point Vector
		for {
			x := rnd2.Float64()*2 - 1
			y := rnd2.Float64()*2 - 1
			if x*x+y*y <= 1 {
				l := center.Sub(n.Origin).Normalize()
				u := l.Cross(RandomUnitVector(rnd2)).Normalize()
				v := l.Cross(u)
				point = u.MulScalar(x * radius).Add(v.MulScalar(y * radius)).Add(center)
				break
			}
		}
		ray := Ray{n.Origin, point.Sub(n.Origin).Normalize()}
		val := 0.0
		// the legacy code compared hit.Shape to the mesh, which never
		// matched because k-d tree hits return the member triangle; here
		// the intended legacy behavior is reproduced faithfully by
		// accepting hits on any triangle of the emissive mesh
		if ray.Direction.Dot(n.Direction) > 0 {
			if hit := scene.Intersect(ray); hit.Ok() && meshOwnsHit(mesh, hit) {
				val = 4.0 * ray.Direction.Dot(n.Direction) * coverage
				accepted++
			}
		}
		// Welford online update
		delta := val - mean
		mean += delta / float64(i+1)
		m2 += delta * (val - mean)
	}
	legacyVar := m2 / k
	if accepted == 0 || accepted == k {
		t.Fatalf("legacy test setup invalid: acceptance %d/%d", accepted, k)
	}
	if areaVar >= legacyVar {
		t.Fatalf("area sampling variance %v should be below legacy %v", areaVar, legacyVar)
	}
	t.Logf("variance: area=%.5f legacy-disk=%.5f (%.1fx reduction), disk acceptance=%.1f%%",
		areaVar, legacyVar, legacyVar/areaVar, 100*float64(accepted)/k)
}

func TestSphereLightBehaviorPreserved(t *testing.T) {
	// sphere lights must keep the original estimator: verify they are not
	// routed through the AreaLight interface and still light the ground
	sphere := NewSphere(V(0, 3, 0), 0.5, LightMaterial(White, 20))
	if _, ok := sphere.(AreaLight); ok {
		t.Fatalf("Sphere must not implement AreaLight")
	}
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(sphere)
	scene.Add(ground)
	rnd := newTestRNG()
	mean, _ := estimateGroundIrradiance(t, &scene, V(0, 0, 0), 20000, rnd, true)
	if mean <= 0 {
		t.Fatalf("sphere light should illuminate ground, got %v", mean)
	}
	// hard shadow mode gives a non-zero, deterministic result
	hard, _ := estimateGroundIrradiance(t, &scene, V(0, 0, 0), 100, rand.New(rand.NewSource(1)), false)
	if hard <= 0 {
		t.Fatalf("sphere hard shadow sampling should still contribute, got %v", hard)
	}
	t.Logf("sphere light: soft=%.4f hard=%.4f", mean, hard)
}

func TestHardShadowTriangleLightEnergyConserved(t *testing.T) {
	// with SoftShadows disabled the centroid is sampled: deterministic.
	// A small light is used because single-point centroid quadrature is
	// exact only in the point-light limit; the area PDF still keeps the
	// estimator unbiased on average over the surface measure.
	mesh := quadMesh(-0.2, 0.2, -0.2, 0.2, 2, 7)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(mesh)
	scene.Add(ground)
	p := V(0, 0, 0)
	analytic := polygonIrradiance(p, V(0, 1, 0), 7,
		[]Vector{V(-0.2, 2, -0.2), V(0.2, 2, -0.2), V(0.2, 2, 0.2), V(-0.2, 2, 0.2)})
	rnd := newTestRNG()
	mean, variance := estimateGroundIrradiance(t, &scene, p, 1000, rnd, false)
	// hard mode samples one fixed centroid, so per-sample values are
	// identical apart from ~1e-15 floating-point jitter in the ray hit
	if variance > 1e-10 {
		t.Fatalf("hard shadow mode must be (numerically) deterministic, variance=%v", variance)
	}
	if math.Abs(mean-analytic)/analytic > 0.02 {
		t.Fatalf("hard shadow energy = %v, analytic = %v", mean, analytic)
	}
	t.Logf("hard shadow: mc=%.4f analytic=%.4f", mean, analytic)
}
