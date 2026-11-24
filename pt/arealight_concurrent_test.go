package pt

import (
	"math/rand"
	"sync"
	"testing"
)

// TestConcurrentAreaLightSampling exercises the area-light sampling path from
// many goroutines, the way the renderer does. It must pass under -race.
func TestConcurrentAreaLightSampling(t *testing.T) {
	meshes := make([]*Mesh, 4)
	for i := range meshes {
		x := float64(i) * 3
		meshes[i] = quadMesh(x, x+2, -1, 1, 3, 5)
	}
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	for _, m := range meshes {
		scene.Add(m)
	}
	scene.Add(ground)
	scene.Compile()
	if len(scene.Lights) != 4 {
		t.Fatalf("expected 4 lights, got %d", len(scene.Lights))
	}

	sampler := NewDirectSampler()
	sampler.SoftShadows = true
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rnd := rand.New(rand.NewSource(seed))
			n := Ray{Origin: V(float64(seed), 0, 0), Direction: V(0, 1, 0)}
			for i := 0; i < 2000; i++ {
				_ = sampler.sampleLights(&scene, n, rnd)
			}
		}(int64(g + 1))
	}
	wg.Wait()
}

// TestTriangleAndMeshLightsCoexist verifies a standalone emissive triangle
// and an emissive mesh can be lights in the same scene and both contribute.
func TestTriangleAndMeshLightsCoexist(t *testing.T) {
	tri := emitTri(V(-3, 3, -1), V(-2, 3, -1), V(-2.5, 3, 1), 6)
	mesh := quadMesh(2, 3, -1, 1, 3, 6)
	ground := NewCube(V(-100, -100, -100), V(100, -0.05, 100), DiffuseMaterial(White))
	scene := Scene{}
	scene.Add(tri)
	scene.Add(mesh)
	scene.Add(ground)
	if len(scene.Lights) != 2 {
		t.Fatalf("expected triangle + mesh = 2 lights, got %d", len(scene.Lights))
	}
	rnd := newTestRNG()
	mean, _ := estimateGroundIrradiance(t, &scene, V(0, 0, 0), 30000, rnd, true)
	if mean <= 0 {
		t.Fatalf("mixed light types must contribute, got %v", mean)
	}
	// disabling one light must reduce the result
	sceneOnlyMesh := Scene{}
	sceneOnlyMesh.Add(mesh)
	sceneOnlyMesh.Add(ground)
	onlyMesh, _ := estimateGroundIrradiance(t, &sceneOnlyMesh, V(0, 0, 0), 30000, rand.New(rand.NewSource(2)), true)
	if onlyMesh >= mean {
		t.Fatalf("removing the triangle light must reduce irradiance: both=%v mesh-only=%v", mean, onlyMesh)
	}
	t.Logf("mixed=%v mesh-only=%v", mean, onlyMesh)
}

// TestPDFZeroIsExplicit checks the documented "pdf == 0 means unsampleable"
// contract across every area light type.
func TestPDFZeroIsExplicit(t *testing.T) {
	rnd := newTestRNG()
	origin := V(0, 0, 1)

	degenerateTri := emitTri(V(0, 0, 0), V(1, 0, 0), V(2, 0, 0), 5)
	if s := degenerateTri.SampleLight(origin, true, rnd); s.PDF != 0 {
		t.Fatalf("degenerate triangle pdf = %v, want 0", s.PDF)
	}

	degenerateMesh := NewMesh([]*Triangle{degenerateTri})
	if s := degenerateMesh.SampleLight(origin, true, rnd); s.PDF != 0 {
		t.Fatalf("degenerate mesh pdf = %v, want 0", s.PDF)
	}

	// a mesh with light area reports a strictly positive, finite pdf
	good := quadMesh(-1, 1, -1, 1, 2, 5)
	s := good.SampleLight(origin, true, rnd)
	if !(s.PDF > 0) || s.PDF != 1/good.LightArea() {
		t.Fatalf("valid mesh pdf = %v, want normalized 1/area", s.PDF)
	}
}
