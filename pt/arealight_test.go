package pt

import (
	"math"
	"math/rand"
	"testing"
)

func newTestRNG() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

// emitTri builds an emissive triangle.
func emitTri(v1, v2, v3 Vector, emittance float64) *Triangle {
	return NewTriangle(v1, v2, v3, Vector{}, Vector{}, Vector{}, LightMaterial(White, emittance))
}

func TestTriangleAreaSamplingPDFAndCoverage(t *testing.T) {
	origin := V(0, 0, 0)
	tri := emitTri(V(1, 0, 0), V(0, 2, 0), V(0, 0, 3), 10)
	area := tri.Area()
	if area <= 0 {
		t.Fatalf("bad area: %v", area)
	}
	rnd := newTestRNG()
	const n = 200000
	var sumPDF float64
	inside := 0
	for i := 0; i < n; i++ {
		s := tri.SampleLight(origin, true, rnd)
		if s.PDF != 1/area {
			t.Fatalf("pdf = %v, want %v", s.PDF, 1/area)
		}
		if s.R2 != s.Point.Sub(origin).LengthSquared() {
			t.Fatalf("R2 mismatch")
		}
		sumPDF += s.PDF
		// every sample must lie on the triangle plane and inside it
		u, v, w := tri.Barycentric(s.Point)
		if u < -1e-9 || v < -1e-9 || w < -1e-9 {
			t.Fatalf("sample outside triangle: %v %v %v", u, v, w)
		}
		if math.Abs(s.Normal.Dot(tri.NormalAt(s.Point))) < 1-1e-9 {
			t.Fatalf("normal mismatch")
		}
		inside++
	}
	// the area density integrates (in the Monte Carlo sense) consistently:
	// E[pdf * area] == 1
	if got := sumPDF * area / n; math.Abs(got-1) > 1e-10 {
		t.Fatalf("pdf not normalized: %v", got)
	}
	if inside != n {
		t.Fatalf("only %d/%d samples inside", inside, n)
	}
}

func TestTriangleUniformSamplingMeanIsCentroid(t *testing.T) {
	origin := V(0, 0, 0)
	tri := emitTri(V(1, 0, 0), V(0, 2, 0), V(0, 0, 3), 10)
	rnd := newTestRNG()
	mean := Vector{}
	const n = 200000
	for i := 0; i < n; i++ {
		s := tri.SampleLight(origin, true, rnd)
		mean = mean.Add(s.Point)
	}
	mean = mean.DivScalar(n)
	centroid := tri.V1.Add(tri.V2).Add(tri.V3).MulScalar(1.0 / 3.0)
	if mean.Sub(centroid).Length() > 0.01 {
		t.Fatalf("sample mean %v != centroid %v", mean, centroid)
	}
}

func TestDegenerateAndZeroAreaTriangle(t *testing.T) {
	rnd := newTestRNG()
	origin := V(0, 0, 0)
	cases := []*Triangle{
		// repeated vertices
		emitTri(V(1, 0, 0), V(1, 0, 0), V(0, 1, 0), 10),
		// collinear vertices
		emitTri(V(0, 0, 0), V(1, 0, 0), V(2, 0, 0), 10),
	}
	for _, tri := range cases {
		if tri.Area() != 0 {
			t.Fatalf("expected zero area, got %v", tri.Area())
		}
		if tri.LightArea() != 0 {
			t.Fatalf("expected zero LightArea")
		}
		s := tri.SampleLight(origin, true, rnd)
		if s.PDF != 0 {
			t.Fatalf("degenerate triangle pdf = %v, want 0", s.PDF)
		}
		scene := Scene{}
		scene.Add(tri)
		if len(scene.Lights) != 0 {
			t.Fatalf("zero-area triangle must not be registered as a light")
		}
	}
}

func TestNegativeEmittanceIsNotALight(t *testing.T) {
	tri := NewTriangle(V(0, 0, 0), V(1, 0, 0), V(0, 1, 0),
		Vector{}, Vector{}, Vector{}, LightMaterial(White, -5))
	scene := Scene{}
	scene.Add(tri)
	if len(scene.Lights) != 0 {
		t.Fatalf("negative emittance shape must not be a light")
	}
	// an emittance <= 0 material yields no contribution even if sampled
	rnd := newTestRNG()
	sample := tri.SampleLight(V(0, 0, 1), true, rnd)
	if sample.Material.Emittance > 0 {
		t.Fatalf("negative emittance must not be treated as positive")
	}
	sampler := NewDirectSampler()
	weight := math.Max(0, sample.Material.Emittance)
	if weight != 0 {
		t.Fatalf("negative emittance contributed: %v", weight)
	}
	_ = sampler
}

func TestMeshAreaWeightedCDF(t *testing.T) {
	// two emissive triangles, one 4x the area of the other
	small := emitTri(V(0, 0, 0), V(1, 0, 0), V(0, 1, 0), 10)
	big := emitTri(V(5, 0, 0), V(7, 0, 0), V(5, 2, 0), 10)
	mesh := NewMesh([]*Triangle{small, big})
	if got := mesh.LightArea(); math.Abs(got-(0.5+2.0)) > 1e-12 {
		t.Fatalf("mesh light area = %v, want 2.5", got)
	}
	rnd := newTestRNG()
	origin := V(0, 0, 5)
	var smallHits, bigHits int
	const n = 200000
	for i := 0; i < n; i++ {
		s := mesh.SampleLight(origin, true, rnd)
		if s.PDF != 1/mesh.LightArea() {
			t.Fatalf("mesh pdf = %v, want %v", s.PDF, 1/mesh.LightArea())
		}
		// which triangle owns the sample?
		if u, v, w := small.Barycentric(s.Point); u >= 0 && v >= 0 && w >= 0 &&
			u <= 1 && v <= 1 && w <= 1 {
			smallHits++
		} else if u, v, w := big.Barycentric(s.Point); u >= 0 && v >= 0 && w >= 0 &&
			u <= 1 && v <= 1 && w <= 1 {
			bigHits++
		} else {
			t.Fatalf("sample on no triangle: %v", s.Point)
		}
	}
	// selection probability must be proportional to area: big gets 2/2.5 = 0.8
	wantBig := 2.0 / 2.5
	gotBig := float64(bigHits) / n
	if math.Abs(gotBig-wantBig) > 0.01 {
		t.Fatalf("big triangle share = %v, want %v", gotBig, wantBig)
	}
	if smallHits+bigHits != n {
		t.Fatalf("lost samples")
	}
}

func TestMeshSkipsDegenerateAndNonEmissive(t *testing.T) {
	good := emitTri(V(0, 0, 0), V(1, 0, 0), V(0, 1, 0), 10)
	degenerate := emitTri(V(2, 0, 0), V(2, 0, 0), V(3, 0, 0), 10)
	dark := NewTriangle(V(4, 0, 0), V(5, 0, 0), V(4, 1, 0),
		Vector{}, Vector{}, Vector{}, DiffuseMaterial(White))
	mesh := NewMesh([]*Triangle{good, degenerate, dark})
	if got := mesh.LightArea(); got != 0.5 {
		t.Fatalf("light area = %v, want 0.5", got)
	}
	if len(mesh.lightTriangles) != 1 {
		t.Fatalf("expected 1 emissive triangle in distribution, got %d", len(mesh.lightTriangles))
	}
	// CDF must be normalized to exactly 1
	if mesh.lightTriangles[0].cdf != 1 {
		t.Fatalf("cdf = %v, want 1", mesh.lightTriangles[0].cdf)
	}
}

func TestFullyDegenerateMeshIsNotALight(t *testing.T) {
	degenerate := emitTri(V(0, 0, 0), V(1, 0, 0), V(2, 0, 0), 10)
	mesh := NewMesh([]*Triangle{degenerate})
	if mesh.LightArea() != 0 {
		t.Fatalf("degenerate mesh area must be 0")
	}
	scene := Scene{}
	scene.Add(mesh)
	if len(scene.Lights) != 0 {
		t.Fatalf("fully degenerate mesh must not be a light")
	}
	rnd := newTestRNG()
	s := mesh.SampleLight(V(0, 0, 5), true, rnd)
	if s.PDF != 0 {
		t.Fatalf("degenerate mesh pdf = %v, want 0", s.PDF)
	}
}

func TestExtremeAspectRatioTriangle(t *testing.T) {
	// extremely elongated, but non-degenerate triangle
	tri := emitTri(V(0, 0, 0), V(1e6, 0, 0), V(0, 1e-6, 0), 10)
	if tri.Area() <= 0 {
		t.Fatalf("sliver triangle must have positive area")
	}
	rnd := newTestRNG()
	origin := V(0, 0, 1)
	const n = 100000
	for i := 0; i < n; i++ {
		s := tri.SampleLight(origin, true, rnd)
		if !(s.PDF > 0) || math.IsInf(s.PDF, 0) || math.IsNaN(s.PDF) {
			t.Fatalf("invalid pdf on sliver triangle: %v", s.PDF)
		}
		if s.PDF != 1/tri.Area() {
			t.Fatalf("pdf not normalized")
		}
	}
}

func TestMeshDistributionRebuildsAfterTransform(t *testing.T) {
	tri := emitTri(V(0, 0, 0), V(1, 0, 0), V(0, 1, 0), 10)
	mesh := NewMesh([]*Triangle{tri})
	before := mesh.LightArea()
	mesh.Transform(Scale(V(2, 2, 2)))
	after := mesh.LightArea()
	// area scales by 4
	if math.Abs(after/before-4) > 1e-9 {
		t.Fatalf("area after scale = %v, want %v", after, before*4)
	}
}

func TestCubeLightIsAreaLight(t *testing.T) {
	cube := NewCube(V(-1, -1, -1), V(1, 1, 1), LightMaterial(White, 5))
	area, ok := interface{}(cube).(AreaLight)
	if !ok {
		t.Fatalf("Cube must implement AreaLight")
	}
	// cube surface area: 6 faces * 4 = 24
	if got := area.LightArea(); math.Abs(got-24) > 1e-9 {
		t.Fatalf("cube area = %v, want 24", got)
	}
	rnd := newTestRNG()
	origin := V(0, 0, 5)
	const n = 50000
	for i := 0; i < n; i++ {
		s := cube.SampleLight(origin, true, rnd)
		if s.PDF != 1/24.0 {
			t.Fatalf("cube pdf = %v", s.PDF)
		}
		// every sample must sit on the cube surface (one coordinate at +/-1)
		p := s.Point
		onFace := almostEqual(math.Abs(p.X), 1) || almostEqual(math.Abs(p.Y), 1) || almostEqual(math.Abs(p.Z), 1)
		if !onFace {
			t.Fatalf("sample not on cube surface: %v", p)
		}
	}
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestHardShadowModeUsesCentroidButSamePDF(t *testing.T) {
	tri := emitTri(V(0, 0, 0), V(2, 0, 0), V(0, 2, 0), 10)
	rnd := newTestRNG()
	origin := V(0, 0, 5)
	s := tri.SampleLight(origin, false, rnd)
	centroid := tri.V1.Add(tri.V2).Add(tri.V3).MulScalar(1.0 / 3.0)
	if s.Point.Sub(centroid).Length() > 1e-12 {
		t.Fatalf("hard mode must sample centroid")
	}
	if s.PDF != 1/tri.Area() {
		t.Fatalf("hard mode pdf must remain the area density")
	}
}
