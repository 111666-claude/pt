package pt

import (
	"math"
	"math/rand"
	"runtime"
	"strings"
	"testing"
)

// robotMesh builds a small asymmetric mesh (a body with a head) so that
// rotations are visually and geometrically distinguishable.
func robotMesh() *Mesh {
	body := NewCube(Vector{-0.4, 0, -0.3}, Vector{0.4, 0.8, 0.3}, DiffuseMaterial(HexColor(0x334455)))
	head := NewCube(Vector{-0.25, 0.8, -0.25}, Vector{0.25, 1.2, 0.25}, DiffuseMaterial(HexColor(0x8899AA)))
	nose := NewCube(Vector{-0.08, 0.9, 0.25}, Vector{0.08, 1.05, 0.4}, DiffuseMaterial(HexColor(0xCC5533)))
	mesh := body.Mesh()
	mesh.Add(head.Mesh())
	mesh.Add(nose.Mesh())
	return mesh
}

// sphereMesh builds a lat/long tessellated sphere with a controllable
// triangle count, used to make memory differences measurable.
func sphereMesh(cols, rows int) *Mesh {
	material := DiffuseMaterial(HexColor(0x7799BB))
	var triangles []*Triangle
	point := func(i, j int) Vector {
		theta := 2 * math.Pi * float64(i) / float64(cols)
		phi := math.Pi * float64(j) / float64(rows)
		return Vector{
			math.Sin(phi) * math.Cos(theta),
			math.Cos(phi),
			math.Sin(phi) * math.Sin(theta),
		}
	}
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			a := point(i, j)
			b := point(i+1, j)
			c := point(i+1, j+1)
			d := point(i, j+1)
			z := Vector{}
			triangles = append(triangles, NewTriangle(a, b, c, z, z, z, material))
			triangles = append(triangles, NewTriangle(a, c, d, z, z, z, material))
		}
	}
	return NewMesh(triangles)
}

func vectorsClose(a, b Vector, eps float64) bool {
	return a.Sub(b).Length() <= eps
}

func TestNewInstanceErrors(t *testing.T) {
	mesh := robotMesh()
	if _, err := NewInstance("x", nil, Identity()); err == nil {
		t.Error("expected error for nil shape")
	}
	if _, err := NewInstance("x", &Mesh{}, Identity()); err == nil {
		t.Error("expected error for empty mesh")
	}
	if _, err := NewInstance("x", mesh, Scale(Vector{0, 1, 1})); err == nil {
		t.Error("expected error for singular matrix")
	}
	nan := Matrix{}
	nan.x00 = math.NaN()
	if _, err := NewInstance("x", mesh, nan); err == nil {
		t.Error("expected error for NaN matrix")
	}
	if _, err := NewInstance("x", mesh, Identity()); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestInstanceIDs(t *testing.T) {
	mesh := robotMesh()
	a, _ := NewInstance("robot-7", mesh, Identity())
	if a.ID != "robot-7" {
		t.Errorf("explicit ID not preserved: %q", a.ID)
	}
	b, _ := NewInstance("", mesh, Identity())
	c, _ := NewInstance("", mesh, Identity())
	if b.ID == "" || c.ID == "" {
		t.Error("auto-generated IDs must be non-empty")
	}
	if b.ID == c.ID {
		t.Errorf("auto-generated IDs must be unique, got %q twice", b.ID)
	}
	// IDs are stable: they do not change across compiles
	id := b.ID
	b.Compile()
	if b.ID != id {
		t.Error("instance ID changed across Compile")
	}
}

func TestInstanceMatchesTransformedShape(t *testing.T) {
	mesh := robotMesh()
	matrix := Identity().Rotate(Vector{0, 1, 0}, 0.7).Rotate(Vector{1, 0, 0}, 0.2).Translate(Vector{1, 2, 3})
	instance := MustInstance("i", mesh, matrix)
	transformed := NewTransformedShape(mesh, matrix)
	instance.Compile()
	transformed.Compile()

	if !boxesClose(instance.BoundingBox(), transformed.BoundingBox(), 1e-9) {
		t.Error("bounding boxes differ from TransformedShape")
	}

	for i := 0; i < 200; i++ {
		rnd := rand.New(rand.NewSource(int64(i)))
		origin := RandomUnitVector(rnd).MulScalar(6).Add(Vector{1, 2, 3})
		ray := Ray{origin, Vector{1, 2, 3}.Sub(origin).Normalize()}
		h1 := instance.Intersect(ray)
		h2 := transformed.Intersect(ray)
		if h1.Ok() != h2.Ok() {
			t.Fatalf("ray %d: hit mismatch %v vs %v", i, h1.Ok(), h2.Ok())
		}
		if !h1.Ok() {
			continue
		}
		if math.Abs(h1.T-h2.T) > 1e-9 {
			t.Errorf("ray %d: T differs: %v vs %v", i, h1.T, h2.T)
		}
		a, b := h1.Info(ray), h2.Info(ray)
		if !vectorsClose(a.Position, b.Position, 1e-9) {
			t.Errorf("ray %d: positions differ: %v vs %v", i, a.Position, b.Position)
		}
		if !vectorsClose(a.Normal, b.Normal, 1e-9) {
			t.Errorf("ray %d: normals differ: %v vs %v", i, a.Normal, b.Normal)
		}
		if a.Inside != b.Inside {
			t.Errorf("ray %d: inside differs", i)
		}
		if a.Material.Color != b.Material.Color {
			t.Errorf("ray %d: materials differ: %v vs %v", i, a.Material.Color, b.Material.Color)
		}
	}
}

func boxesClose(a, b Box, eps float64) bool {
	return vectorsClose(a.Min, b.Min, eps) && vectorsClose(a.Max, b.Max, eps)
}

func TestInstanceMatchesBakedCopy(t *testing.T) {
	mesh := robotMesh()
	matrix := Identity().Rotate(Vector{0, 1, 0}, 1.1).Translate(Vector{-3, 0.5, 2})
	instance := MustInstance("i", mesh, matrix)
	baked := mesh.Copy()
	baked.Transform(matrix)
	instance.Compile()
	baked.Compile()

	// the instance box transforms the shared mesh's box (TransformedShape
	// semantics), so it is a tight superset of the baked copy's box
	ib, bb := instance.BoundingBox(), baked.BoundingBox()
	if !ib.Contains(bb.Min) || !ib.Contains(bb.Max) {
		t.Error("instance bounding box does not enclose baked copy box")
	}
	if !boxesClose(ib, NewTransformedShape(mesh, matrix).BoundingBox(), 1e-12) {
		t.Error("instance bounding box differs from TransformedShape box")
	}
	for i := 0; i < 200; i++ {
		rnd := rand.New(rand.NewSource(int64(i) * 13))
		origin := RandomUnitVector(rnd).MulScalar(8).Add(Vector{-3, 0.5, 2})
		// jitter the target: aiming exactly at the mesh center produces
		// degenerate grazing rays where ulp-level differences between the
		// two pipelines legitimately select different triangles
		target := RandomUnitVector(rnd).MulScalar(0.2).Add(Vector{-3, 0.5, 2})
		ray := Ray{origin, target.Sub(origin).Normalize()}
		h1 := instance.Intersect(ray)
		h2 := baked.Intersect(ray)
		if h1.Ok() != h2.Ok() {
			t.Fatalf("ray %d: hit mismatch %v vs %v", i, h1.Ok(), h2.Ok())
		}
		if !h1.Ok() {
			continue
		}
		a, b := h1.Info(ray), h2.Info(ray)
		if !vectorsClose(a.Position, b.Position, 1e-7) {
			t.Errorf("ray %d: positions differ: %v vs %v", i, a.Position, b.Position)
		}
		if !vectorsClose(a.Normal, b.Normal, 1e-7) {
			t.Errorf("ray %d: normals differ: %v vs %v", i, a.Normal, b.Normal)
		}
	}
}

func TestInstanceUVNormalMaterial(t *testing.T) {
	material := GlossyMaterial(HexColor(0x226688), 1.5, 0.3)
	tri := NewTriangle(
		Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{0, 1, 0},
		Vector{0, 0, 0}, Vector{1, 0, 0}, Vector{0, 1, 0},
		material)
	matrix := Identity().Rotate(Vector{0, 1, 0}, 0.9).Translate(Vector{5, 1, -2})
	instance := MustInstance("tri", tri, matrix)

	// world-space point on the transformed triangle
	local := Vector{0.25, 0.25, 0}
	world := matrix.MulPosition(local)

	if uv := instance.UV(world); !vectorsClose(uv, tri.UV(local), 1e-9) {
		t.Errorf("UV does not pass through transform: %v", uv)
	}
	wantNormal := matrix.MulDirection(Vector{0, 0, 1})
	if n := instance.NormalAt(world); !vectorsClose(n, wantNormal, 1e-9) {
		t.Errorf("NormalAt does not pass through transform: %v, want %v", n, wantNormal)
	}
	if m := instance.MaterialAt(world); m.Color != material.Color || m.Gloss != material.Gloss {
		t.Errorf("MaterialAt does not delegate to shared shape: %v", m)
	}

	// material override
	override := DiffuseMaterial(HexColor(0xFF0000))
	instance.SetMaterial(override)
	if m := instance.MaterialAt(world); m.Color != override.Color {
		t.Errorf("material override not applied: %v", m)
	}
	ray := Ray{world.Add(Vector{0, 0, 1}), Vector{0, 0, -1}}
	hit := instance.Intersect(ray)
	if !hit.Ok() {
		t.Fatal("expected hit")
	}
	if info := hit.Info(ray); info.Material.Color != override.Color {
		t.Errorf("override not applied in Intersect: %v", info.Material.Color)
	}
	// the shared shape is unaffected
	if tri.MaterialAt(local).Color != material.Color {
		t.Error("override leaked into shared shape")
	}
	instance.ClearMaterial()
	if m := instance.MaterialAt(world); m.Color != material.Color {
		t.Error("ClearMaterial did not restore shared material")
	}
}

func TestInstanceNesting(t *testing.T) {
	mesh := robotMesh()
	m1 := Identity().Rotate(Vector{0, 1, 0}, 0.5).Translate(Vector{1, 0, 0})
	m2 := Identity().Rotate(Vector{1, 0, 0}, 0.3).Translate(Vector{0, 2, 1})
	inner := MustInstance("inner", mesh, m1)
	outer := MustInstance("outer", inner, m2)
	flat := MustInstance("flat", mesh, m2.Mul(m1))
	outer.Compile()

	for i := 0; i < 100; i++ {
		rnd := rand.New(rand.NewSource(int64(i) * 31))
		origin := RandomUnitVector(rnd).MulScalar(9).Add(Vector{1, 2, 1})
		target := RandomUnitVector(rnd).MulScalar(0.2).Add(Vector{1, 2, 1})
		ray := Ray{origin, target.Sub(origin).Normalize()}
		h1 := outer.Intersect(ray)
		h2 := flat.Intersect(ray)
		if h1.Ok() != h2.Ok() {
			t.Fatalf("ray %d: nested hit mismatch %v vs %v", i, h1.Ok(), h2.Ok())
		}
		if !h1.Ok() {
			continue
		}
		a, b := h1.Info(ray), h2.Info(ray)
		if !vectorsClose(a.Position, b.Position, 1e-8) {
			t.Errorf("ray %d: nested positions differ: %v vs %v", i, a.Position, b.Position)
		}
		if !vectorsClose(a.Normal, b.Normal, 1e-8) {
			t.Errorf("ray %d: nested normals differ: %v vs %v", i, a.Normal, b.Normal)
		}
	}

	// instance nested inside a plain TransformedShape
	wrapped := NewTransformedShape(inner, m2)
	wrapped.Compile()
	for i := 0; i < 100; i++ {
		rnd := rand.New(rand.NewSource(int64(i) * 31))
		origin := RandomUnitVector(rnd).MulScalar(9).Add(Vector{1, 2, 1})
		target := RandomUnitVector(rnd).MulScalar(0.2).Add(Vector{1, 2, 1})
		ray := Ray{origin, target.Sub(origin).Normalize()}
		h1 := wrapped.Intersect(ray)
		h2 := flat.Intersect(ray)
		if h1.Ok() != h2.Ok() {
			t.Fatalf("ray %d: wrapped hit mismatch %v vs %v", i, h1.Ok(), h2.Ok())
		}
	}
}

func TestInstanceValidation(t *testing.T) {
	mesh := robotMesh()

	// self cycle
	a := MustInstance("a", mesh, Identity())
	a.Shape = a
	if err := a.Validate(); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Errorf("expected cycle error, got %v", err)
	}

	// two-node cycle
	b := MustInstance("b", mesh, Identity())
	c := MustInstance("c", b, Identity())
	b.Shape = c
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Errorf("expected cycle error, got %v", err)
	}

	// nil shape introduced after construction
	d := MustInstance("d", mesh, Identity())
	d.Shape = nil
	if err := d.Validate(); err == nil {
		t.Error("expected error for nil shape")
	}

	// singular matrix introduced after construction
	e := MustInstance("e", mesh, Identity())
	e.Matrix = Scale(Vector{1, 0, 1})
	if err := e.Validate(); err == nil {
		t.Error("expected error for singular matrix")
	}

	// depth limit
	depth := MaxInstanceDepth
	defer func() { MaxInstanceDepth = depth }()
	MaxInstanceDepth = 3
	chain := MustInstance("c0", mesh, Identity())
	for i := 1; i <= 5; i++ {
		chain = MustInstance("c"+string(rune('0'+i)), chain, Identity())
	}
	if err := chain.Validate(); err == nil || !strings.Contains(err.Error(), "depth") {
		t.Errorf("expected depth error, got %v", err)
	}
	MaxInstanceDepth = depth

	// count limit
	count := MaxInstanceCount
	defer func() { MaxInstanceCount = count }()
	MaxInstanceCount = 2
	if err := chain.Validate(); err == nil || !strings.Contains(err.Error(), "graph size") {
		t.Errorf("expected count error, got %v", err)
	}
	MaxInstanceCount = count

	// shared sub-instance (DAG, not a cycle) is valid
	shared := MustInstance("shared", mesh, Identity())
	p := MustInstance("p", shared, Identity())
	q := MustInstance("q", p, Identity())
	r := MustInstance("r", p, Identity())
	top := MustInstance("top", q, Identity())
	_ = r
	if err := top.Validate(); err != nil {
		t.Errorf("DAG should be valid: %v", err)
	}
}

func TestInstanceCompilePanicsOnInvalidGraph(t *testing.T) {
	mesh := robotMesh()
	a := MustInstance("a", mesh, Identity())
	a.Shape = a
	defer func() {
		if recover() == nil {
			t.Error("expected Compile to panic on cyclic graph")
		}
	}()
	a.Compile()
}

func TestSharedMeshModification(t *testing.T) {
	mesh := robotMesh()
	instance := MustInstance("i", mesh, Identity())
	scene := &Scene{}
	scene.Add(instance)
	scene.Compile()

	box1 := instance.BoundingBox()
	ray := Ray{Vector{0.2, 0.5, 5}, Vector{0, 0, -1}}
	if hit := scene.Intersect(ray); !hit.Ok() {
		t.Fatal("expected hit before modification")
	}

	// modifying the shared mesh is visible through the instance: no vertex
	// data was copied, so nothing goes stale. The instance tracks the new
	// geometry after recompiling. (As with any geometry moved directly in
	// stock pt, a Scene's already-built k-d tree is not rebuilt, so a new
	// scene is needed to ray-trace the moved geometry.)
	mesh.Transform(Translate(Vector{10, 0, 0}))
	instance.Compile()
	box2 := instance.BoundingBox()
	if boxesClose(box1, box2, 1e-9) {
		t.Error("bounding box did not track shared mesh modification")
	}
	if hit := instance.Intersect(ray); hit.Ok() {
		t.Error("expected miss at old location after modification")
	}
	moved := Ray{Vector{10.2, 0.5, 5}, Vector{0, 0, -1}}
	if hit := instance.Intersect(moved); !hit.Ok() {
		t.Error("expected hit at new location after modification")
	}
	fresh := &Scene{}
	fresh.Add(instance)
	fresh.Compile()
	if hit := fresh.Intersect(moved); !hit.Ok() {
		t.Error("expected hit at new location in rebuilt scene")
	}
}

func TestInstanceCompileSharesGeometry(t *testing.T) {
	mesh := robotMesh()
	first := &mesh.Triangles[0]
	scene := &Scene{}
	for i := 0; i < 100; i++ {
		m := Identity().Rotate(Vector{0, 1, 0}, float64(i)).Translate(Vector{float64(i), 0, 0})
		scene.Add(MustInstance("", mesh, m))
	}
	scene.Compile()
	if len(mesh.Triangles) == 0 || &mesh.Triangles[0] != first {
		t.Error("compiling instances copied triangle data")
	}
	if mesh.tree == nil {
		t.Error("shared mesh was not compiled")
	}
}

func heapAlloc() uint64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.HeapAlloc
}

func TestInstanceMemory(t *testing.T) {
	const k = 500
	mesh := sphereMesh(16, 8) // 448 triangles
	transform := func(i int) Matrix {
		x := float64(i%25) * 3
		z := float64(i/25) * 3
		return Identity().Rotate(Vector{0, 1, 0}, float64(i)*0.3).Translate(Vector{x, 0, z})
	}

	before := heapAlloc()
	copied := &Scene{}
	for i := 0; i < k; i++ {
		c := mesh.Copy()
		c.Transform(transform(i))
		copied.Add(c)
	}
	copied.Compile()
	copiedMem := heapAlloc() - before
	runtime.KeepAlive(copied)

	before = heapAlloc()
	instanced := &Scene{}
	for i := 0; i < k; i++ {
		instanced.Add(MustInstance("", mesh, transform(i)))
	}
	instanced.Compile()
	instanceMem := heapAlloc() - before
	runtime.KeepAlive(instanced)

	t.Logf("copied scene: %d bytes, instanced scene: %d bytes (%.1fx less)",
		copiedMem, instanceMem, float64(copiedMem)/float64(instanceMem))
	if instanceMem*4 >= copiedMem {
		t.Errorf("instancing saved too little memory: copied=%d instanced=%d", copiedMem, instanceMem)
	}
}

func renderScene(scene *Scene, camera *Camera, w, h int) []Color {
	sampler := &DefaultSampler{4, 2, true, true, LightModeAll, SpecularModeNaive}
	scene.Compile()
	result := make([]Color, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// identical random sequence for every pixel in both scenes
			rnd := rand.New(rand.NewSource(int64(y*w+x)*7919 + 42))
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rnd)
			result[y*w+x] = sampler.Sample(scene, ray, rnd)
		}
	}
	return result
}

func TestInstanceRenderEquivalence(t *testing.T) {
	const k = 100
	mesh := robotMesh()
	override := GlossyMaterial(HexColor(0xAA3322), 1.4, 0.2)
	transform := func(i int) Matrix {
		x := float64(i%10)*1.6 - 7.2
		z := float64(i/10)*1.6 - 7.2
		return Identity().Rotate(Vector{0, 1, 0}, float64(i)*0.37).Translate(Vector{x, 0, z})
	}

	instanced := &Scene{}
	instanced.Add(NewSphere(Vector{4, 10, 8}, 1.5, LightMaterial(White, 4)))
	for i := 0; i < k; i++ {
		inst := MustInstance("", mesh, transform(i))
		if i%3 == 0 {
			inst.SetMaterial(override)
		}
		instanced.Add(inst)
	}

	copied := &Scene{}
	copied.Add(NewSphere(Vector{4, 10, 8}, 1.5, LightMaterial(White, 4)))
	for i := 0; i < k; i++ {
		c := mesh.Copy()
		c.Transform(transform(i))
		if i%3 == 0 {
			c.SetMaterial(override)
		}
		copied.Add(c)
	}

	camera := LookAt(Vector{0, 9, 16}, Vector{0, 0.5, 0}, Vector{0, 1, 0}, 40)
	const w, h = 48, 48
	a := renderScene(instanced, &camera, w, h)
	b := renderScene(copied, &camera, w, h)

	maxDiff := 0.0
	identical := 0
	for i := range a {
		d := a[i].Sub(b[i]).MaxComponent()
		if d < 0 {
			d = -d
		}
		if d == 0 {
			identical++
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("pixels: %d/%d identical, max per-pixel difference %.3g", identical, len(a), maxDiff)
	if maxDiff > 1e-6 {
		t.Errorf("instanced render differs from copied render by %g", maxDiff)
	}
}
