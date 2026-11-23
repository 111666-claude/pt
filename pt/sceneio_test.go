package pt

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

func ptrFloat(x float64) *float64 { return &x }
func ptrInt(x int) *int           { return &x }
func ptrBool(x bool) *bool        { return &x }
func ptrString(x string) *string  { return &x }

// writePNGTexture creates a small deterministic PNG for texture tests.
func writePNGTexture(t *testing.T, path string, w, h int, fn func(x, y int) color.Color) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, fn(x, y))
		}
	}
	if err := SavePNG(path, img); err != nil {
		t.Fatalf("save texture: %v", err)
	}
}

// writeOBJ writes a minimal OBJ file containing a single triangle.
func writeOBJ(t *testing.T, path string) {
	t.Helper()
	content := "o tri\n" +
		"v 0 0 0\nv 1 0 0\nv 0 1 0\n" +
		"vt 0 0\nvt 1 0\nvt 0 1\n" +
		"vn 0 0 1\n" +
		"f 1/1/1 2/2/1 3/3/1\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write obj: %v", err)
	}
}

// writeSTL writes a minimal binary STL containing a single triangle. The
// on-disk record is 50 bytes: 12 float32 values (normal + 3 vertices) plus
// a uint16 attribute byte count.
func writeSTL(t *testing.T, path string, v1, v2, v3 Vector) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create stl: %v", err)
	}
	defer f.Close()
	header := make([]byte, 80)
	if _, err := f.Write(header); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte{1, 0, 0, 0}); err != nil { // uint32 count = 1
		t.Fatal(err)
	}
	writeF32 := func(x float64) {
		bits := math.Float32bits(float32(x))
		var b [4]byte
		b[0] = byte(bits)
		b[1] = byte(bits >> 8)
		b[2] = byte(bits >> 16)
		b[3] = byte(bits >> 24)
		if _, err := f.Write(b[:]); err != nil {
			t.Fatal(err)
		}
	}
	for _, v := range []Vector{{0, 1, 0}, v1, v2, v3} {
		writeF32(v.X)
		writeF32(v.Y)
		writeF32(v.Z)
	}
	if _, err := f.Write([]byte{0, 0}); err != nil { // attribute byte count
		t.Fatal(err)
	}
}

// smallSceneDocument exercises every section of the schema: camera with
// focus, sampler, background texture, two materials with texture slots, a
// file mesh, an inline mesh, all four primitives, a transform and a
// reference.
func smallSceneDocument() *SceneDocument {
	rot := matrixFromMatrix(Rotate(V(0, 0, 1), Radians(30)))
	return &SceneDocument{
		Version: SceneSchemaVersion,
		Camera: &CameraDocument{
			Type:    "basis",
			Origin:  V(3, 2, 2),
			Right:   V(1, 0, 0),
			Up:      V(0, 1, 0),
			Forward: V(0, 0, -1),
			M:       1 / math.Tan(Radians(40)/2),
			Focus: &FocusDocument{
				FocalPoint:     V(0, 0, 0),
				ApertureRadius: 0.01,
			},
		},
		Sampler: &SamplerDocument{
			Type: "default", FirstHitSamples: ptrInt(4), MaxBounces: ptrInt(3),
			DirectLighting: ptrBool(true), SoftShadows: ptrBool(true),
			LightMode: ptrString("all"), SpecularMode: ptrString("first"),
		},
		Background: &BackgroundDocument{Texture: "tex-env", TextureAngle: ptrFloat(0.5)},
		Materials: []MaterialDocument{
			{
				ID: "mat-floor", Color: &Color{0.8, 0.8, 0.7},
				BumpMultiplier: ptrFloat(1), Emittance: ptrFloat(0), Index: ptrFloat(1),
				Gloss: ptrFloat(0), Tint: ptrFloat(0), Reflectivity: ptrFloat(-1),
				Transparent: ptrBool(false), Texture: "tex-checker",
			},
			{
				ID: "mat-light", Color: &White, Emittance: ptrFloat(50),
				BumpMultiplier: ptrFloat(1), Index: ptrFloat(1),
				Gloss: ptrFloat(0), Tint: ptrFloat(0), Reflectivity: ptrFloat(-1),
				Transparent: ptrBool(false),
			},
		},
		Textures: []TextureDocument{
			{ID: "tex-env", Path: "env.png"},
			{ID: "tex-checker", Path: "checker.png", Pow: ptrFloat(1 / 2.2)},
		},
		Meshes: []MeshDocument{
			{ID: "mesh-file", Path: "tri.obj", Material: "mat-floor"},
			{ID: "mesh-inline", Triangles: []TriangleDocument{{
				V1: V(0, 0, 0), V2: V(1, 0, 0), V3: V(0, 1, 0),
				N1: V(0, 0, 1), N2: V(0, 0, 1), N3: V(0, 0, 1),
				T1: V(0, 0, 0), T2: V(1, 0, 0), T3: V(0, 1, 0),
				Material: "mat-floor",
			}}},
		},
		Shapes: []ShapeDocument{
			{ID: "floor", Type: "plane", Material: "mat-floor",
				Point: ptrVec(V(0, 0, 0)), Normal: ptrVec(V(0, 0, 1))},
			{ID: "ball", Type: "sphere", Material: "mat-floor",
				Center: ptrVec(V(0, 0, 1)), Radius: ptrFloat(1)},
			{Type: "cube", Material: "mat-floor",
				Min: ptrVec(V(-2, -2, 1)), Max: ptrVec(V(-1, -1, 2))},
			{Type: "cylinder", Material: "mat-floor",
				Radius: ptrFloat(0.5), Z0: ptrFloat(0), Z1: ptrFloat(2)},
			{Type: "mesh", Material: "mat-floor", Mesh: "mesh-file",
				Matrix: &rot},
			{Type: "mesh", Material: "mat-floor", Mesh: "mesh-inline"},
			{ID: "lamp", Type: "sphere", Material: "mat-light",
				Center: ptrVec(V(3, 3, 5)), Radius: ptrFloat(0.7)},
			{Type: "reference", Ref: "ball", Matrix: &rot},
		},
	}
}

func ptrVec(v Vector) *Vector { return &v }

func writeSceneResources(t *testing.T, dir string) {
	t.Helper()
	writePNGTexture(t, filepath.Join(dir, "env.png"), 8, 4, func(x, y int) color.Color {
		return color.RGBA{uint8(x * 30), uint8(y * 60), 128, 255}
	})
	writePNGTexture(t, filepath.Join(dir, "checker.png"), 4, 4, func(x, y int) color.Color {
		if (x+y)%2 == 0 {
			return color.RGBA{255, 255, 255, 255}
		}
		return color.RGBA{0, 0, 0, 255}
	})
	writeOBJ(t, filepath.Join(dir, "tri.obj"))
	writeSTL(t, filepath.Join(dir, "tri.stl"),
		V(0, 0, 0), V(1, 0, 0), V(0, 1, 0))
}

// ---------------------------------------------------------------------------
// round trip + stable bytes
// ---------------------------------------------------------------------------

func TestMarshalUnmarshalRoundTrip(t *testing.T) {
	doc := smallSceneDocument()
	data, err := MarshalSceneDocument(doc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	back, err := UnmarshalSceneDocument(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	data2, err := MarshalSceneDocument(back)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	if !bytes.Equal(data, data2) {
		t.Fatalf("serialization is not stable across a round trip\nfirst:\n%s\nsecond:\n%s", data, data2)
	}
	if back.Version != SceneSchemaVersion {
		t.Fatalf("version = %d, want %d", back.Version, SceneSchemaVersion)
	}
}

func TestGoldenBytes(t *testing.T) {
	doc := smallSceneDocument()
	data, err := MarshalSceneDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	golden, err := os.ReadFile(filepath.Join("testdata", "golden.ptscene.json"))
	if err != nil {
		t.Fatalf("read golden fixture: %v", err)
	}
	if !bytes.Equal(data, golden) {
		t.Fatalf("output does not match golden fixture\n--- got ---\n%s\n--- want ---\n%s", data, golden)
	}
	// The golden file must itself reload without error.
	if _, err := UnmarshalSceneDocument(golden); err != nil {
		t.Fatalf("golden fixture fails validation: %v", err)
	}
}

func TestLoadRoundTripCompiles(t *testing.T) {
	dir := t.TempDir()
	writeSceneResources(t, dir)
	path := filepath.Join(dir, "scene.ptscene.json")
	doc := smallSceneDocument()
	if err := SaveSceneDocument(path, doc); err != nil {
		t.Fatal(err)
	}
	compiled, err := LoadScene(path, nil)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if compiled.Camera == nil || compiled.Sampler == nil || compiled.Scene == nil {
		t.Fatal("compiled scene missing components")
	}
	// 7 contributing nodes: the plain transformed reference adds one node;
	// see buildShapes reference semantics.
	if got := len(compiled.Scene.Shapes); got != 8 {
		t.Fatalf("shape count = %d, want 8", got)
	}
	if len(compiled.Scene.Lights) != 1 {
		t.Fatalf("light count = %d, want 1", len(compiled.Scene.Lights))
	}
	compiled.Scene.Compile()
	if !compiled.Camera.IsFinite() {
		t.Fatal("camera non-finite")
	}
}

func TestSaveAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "scene.ptscene.json")
	bad := &SceneDocument{
		Version: SceneSchemaVersion,
		Shapes:  []ShapeDocument{{Type: "sphere" /* missing center/radius */}},
	}
	if err := SaveSceneDocument(path, bad); err == nil {
		t.Fatal("expected validation error, got nil")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("no file should remain after failed save, stat err=%v", err)
	}
}

// ---------------------------------------------------------------------------
// unknown fields, defaults and version policy
// ---------------------------------------------------------------------------

func TestUnknownFieldsIgnored(t *testing.T) {
	doc := smallSceneDocument()
	data, err := MarshalSceneDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	raw["futureExtension"] = map[string]interface{}{"nested": []int{1, 2, 3}}
	raw["shapes"].([]interface{})[0].(map[string]interface{})["experimental"] = 42
	extended, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := UnmarshalSceneDocument(extended); err != nil {
		t.Fatalf("unknown fields must be ignored: %v", err)
	}
}

func TestDefaultsApplied(t *testing.T) {
	dir := t.TempDir()
	writePNGTexture(t, filepath.Join(dir, "env.png"), 2, 2, func(x, y int) color.Color {
		return color.RGBA{10, 20, 30, 255}
	})
	minimal := `{
  "version": 1,
  "background": {"texture": "tex-env"},
  "textures": [{"id": "tex-env", "path": "env.png"}],
  "materials": [{"id": "m1"}],
  "shapes": [
    {"type": "plane", "material": "m1", "point": [0,0,0], "normal": [0,0,1]},
    {"type": "sphere", "material": "m1", "center": [0,0,1], "radius": 1}
  ]
}`
	path := filepath.Join(dir, "min.json")
	if err := os.WriteFile(path, []byte(minimal), 0644); err != nil {
		t.Fatal(err)
	}
	compiled, err := LoadScene(path, nil)
	if err != nil {
		t.Fatalf("load minimal: %v", err)
	}
	if compiled.Camera != nil || compiled.Sampler != nil {
		t.Fatal("optional sections must default to nil")
	}
	m := compiled.Document.Materials[0]
	// omitted optional fields stay nil in the DTO (preserve-defaults policy)
	if m.BumpMultiplier != nil {
		t.Fatalf("omitted field must stay nil, got %v", *m.BumpMultiplier)
	}
	// ... while the compiled material receives the documented default
	compiled.Scene.Compile()
	plane := compiled.Scene.Shapes[0]
	mat := plane.MaterialAt(V(0, 0, 0))
	if mat.BumpMultiplier != 1 {
		t.Fatalf("compiled bump multiplier = %v, want 1", mat.BumpMultiplier)
	}
	if compiled.Scene.Texture == nil {
		t.Fatal("background texture not resolved")
	}
	if len(compiled.Scene.Shapes) != 2 {
		t.Fatalf("shape count = %d", len(compiled.Scene.Shapes))
	}
}

func TestVersionPolicy(t *testing.T) {
	// missing version -> treated as v1
	doc, err := UnmarshalSceneDocument([]byte(`{"materials":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	if doc.Version != 1 {
		t.Fatalf("missing version should default to 1, got %d", doc.Version)
	}
	// future version rejected with a reason
	_, err = UnmarshalSceneDocument([]byte(`{"version": 99}`))
	if err == nil || !strings.Contains(err.Error(), "version") {
		t.Fatalf("future version must be rejected, got %v", err)
	}
	// migration entry point
	migrated, err := MigrateSceneDocument(&SceneDocument{Version: 1})
	if err != nil || migrated.Version != 1 {
		t.Fatalf("v1 migration: %v %+v", err, migrated)
	}
	if _, err := MigrateSceneDocument(&SceneDocument{Version: 42}); err == nil {
		t.Fatal("migration of unknown version must fail")
	}
}

// ---------------------------------------------------------------------------
// bad input fixtures
// ---------------------------------------------------------------------------

func TestBadInputs(t *testing.T) {
	dir := t.TempDir()
	writeSceneResources(t, dir)
	good := smallSceneDocument()
	cases := []struct {
		name   string
		mutate func(d *SceneDocument)
		path   string // expected substring in the contextual error
	}{
		{"missing texture file", func(d *SceneDocument) {
			d.Textures[0].Path = "nope.png"
		}, "textures[0]"},
		{"missing mesh file", func(d *SceneDocument) {
			d.Meshes[0].Path = "nope.stl"
		}, "meshes[0]"},
		{"absolute texture path", func(d *SceneDocument) {
			d.Textures[0].Path = "/etc/passwd"
		}, "absolute paths"},
		{"path traversal", func(d *SceneDocument) {
			d.Textures[0].Path = "../secret.png"
		}, "escapes the scene root"},
		{"duplicate material id", func(d *SceneDocument) {
			d.Materials = append(d.Materials, d.Materials[0])
		}, "duplicate material id"},
		{"duplicate shape id", func(d *SceneDocument) {
			d.Shapes = append(d.Shapes, ShapeDocument{ID: "ball", Type: "sphere",
				Material: "mat-floor", Center: ptrVec(V(0, 0, 0)), Radius: ptrFloat(1)})
		}, "duplicate shape id"},
		{"unknown material ref", func(d *SceneDocument) {
			d.Shapes[1].Material = "ghost"
		}, "unknown material id"},
		{"unknown texture ref", func(d *SceneDocument) {
			d.Materials[0].Texture = "ghost"
		}, "unknown texture id"},
		{"unknown shape ref", func(d *SceneDocument) {
			d.Shapes[len(d.Shapes)-1].Ref = "ghost"
		}, "unknown shape id"},
		{"non-finite vector", func(d *SceneDocument) {
			d.Shapes[1].Center.X = math.NaN()
		}, "non-finite"},
		{"non-finite color", func(d *SceneDocument) {
			d.Materials[0].Color.R = math.Inf(1)
		}, "non-finite"},
		{"singular matrix", func(d *SceneDocument) {
			z := Matrix4{}
			d.Shapes[4].Matrix = &z
		}, "singular"},
		{"negative radius", func(d *SceneDocument) {
			d.Shapes[1].Radius = ptrFloat(-1)
		}, "radius"},
		{"degenerate cube", func(d *SceneDocument) {
			d.Shapes[2].Max = d.Shapes[2].Min
		}, "cube"},
		{"zero plane normal", func(d *SceneDocument) {
			d.Shapes[0].Normal = ptrVec(V(0, 0, 0))
		}, "normal"},
		{"bad sampler enum", func(d *SceneDocument) {
			d.Sampler.LightMode = ptrString("moonlight")
		}, "lightMode"},
		{"non-square samples", func(d *SceneDocument) {
			d.Sampler.FirstHitSamples = ptrInt(7)
		}, "perfect square"},
		{"bad fovy", func(d *SceneDocument) {
			d.Camera = &CameraDocument{Type: "lookAt",
				Eye: ptrVec(V(0, 0, 0)), Center: ptrVec(V(0, 0, -1)),
				WorldUp: ptrVec(V(0, 1, 0)), Fovy: ptrFloat(0)}
		}, "fovy"},
		{"degenerate inline triangle", func(d *SceneDocument) {
			d.Meshes[1].Triangles[0].V2 = d.Meshes[1].Triangles[0].V1
		}, "degenerate triangle"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := deepCopyDocument(t, good)
			tc.mutate(&d)
			path := filepath.Join(dir, "bad-"+tc.name+".json")
			data, err := MarshalSceneDocument(&d)
			if err == nil {
				err = os.WriteFile(path, data, 0644)
			}
			if err == nil {
				_, err = LoadScene(path, nil)
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.path)
			}
			if !strings.Contains(err.Error(), tc.path) {
				t.Fatalf("error %q does not mention %q", err.Error(), tc.path)
			}
			var se *SchemaError
			if !asSchemaError(err, &se) {
				t.Fatalf("error should be a *SchemaError, got %T", err)
			}
		})
	}
}

// deepCopyDocument round-trips a document through JSON so every test case
// starts from an isolated copy (mutations never leak between cases).
func deepCopyDocument(t *testing.T, src *SceneDocument) SceneDocument {
	t.Helper()
	data, err := MarshalSceneDocument(src)
	if err != nil {
		t.Fatal(err)
	}
	dst, err := UnmarshalSceneDocument(data)
	if err != nil {
		t.Fatal(err)
	}
	return *dst
}

func asSchemaError(err error, target **SchemaError) bool {
	for err != nil {
		if se, ok := err.(*SchemaError); ok {
			*target = se
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

func TestCircularReference(t *testing.T) {
	doc := &SceneDocument{
		Version:   SceneSchemaVersion,
		Materials: []MaterialDocument{{ID: "m", Color: &White}},
		Shapes: []ShapeDocument{
			{ID: "a", Type: "reference", Ref: "b"},
			{ID: "b", Type: "reference", Ref: "a"},
		},
	}
	_, err := UnmarshalSceneDocument(mustMarshal(t, doc))
	if err == nil || !strings.Contains(err.Error(), "circular") {
		t.Fatalf("expected circular reference error, got %v", err)
	}
}

func TestOversizedArraysRejected(t *testing.T) {
	doc := &SceneDocument{
		Version:   SceneSchemaVersion,
		Materials: make([]MaterialDocument, 4),
	}
	for i := range doc.Materials {
		doc.Materials[i] = MaterialDocument{ID: string(rune('a' + i))}
	}
	data := mustMarshal(t, doc)
	dir := t.TempDir()
	path := filepath.Join(dir, "oversized.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadScene(path, &LoadOptions{MaxMaterials: 3})
	if err == nil || !strings.Contains(err.Error(), "too many materials") {
		t.Fatalf("expected array limit error, got %v", err)
	}
	// inline triangle budget: validate with a cap of zero, which the
	// single-triangle fixture must exceed.
	doc3 := &SceneDocument{
		Version:   SceneSchemaVersion,
		Materials: []MaterialDocument{{ID: "m", Color: &White}},
		Meshes: []MeshDocument{{
			ID: "tiny",
			Triangles: []TriangleDocument{{
				V1: V(0, 0, 0), V2: V(1, 0, 0), V3: V(0, 1, 0), Material: "m",
			}},
		}},
	}
	zeroCap := LoadOptions{}.limits()
	zeroCap.MaxTrianglesPerMesh = 0
	if err := doc3.validateWithLimits(zeroCap); err == nil ||
		!strings.Contains(err.Error(), "triangles") {
		t.Fatalf("expected triangle limit error, got %v", err)
	}
	// raw document size guard
	big := bytes.Repeat([]byte(" "), int(DefaultMaxDocumentBytes)+1)
	if _, err := UnmarshalSceneDocument(big); err == nil {
		t.Fatal("expected error for oversized/blank document")
	}
}

// ---------------------------------------------------------------------------
// failure must not pollute an existing scene or global caches
// ---------------------------------------------------------------------------

func TestFailureIsAtomic(t *testing.T) {
	dir := t.TempDir()
	writeSceneResources(t, dir)
	doc := smallSceneDocument()
	doc.Textures[0].Path = "missing.png"
	data, err := MarshalSceneDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "broken.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	before := len(textures)
	compiled, err := LoadScene(path, nil)
	if err == nil {
		t.Fatal("expected load failure")
	}
	if compiled != nil {
		t.Fatal("failed load must return nil compiled scene")
	}
	if len(textures) != before {
		t.Fatalf("global texture cache mutated on failure: before=%d after=%d",
			before, len(textures))
	}
}

// ---------------------------------------------------------------------------
// rendering equivalence: Go-built scene vs JSON-loaded scene
// ---------------------------------------------------------------------------

func renderSceneToColors(t *testing.T, scene *Scene, camera *Camera, sampler Sampler) []Color {
	t.Helper()
	scene.Compile()
	const w, h = 12, 12
	rnd := rand.New(rand.NewSource(42))
	out := make([]Color, 0, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ray := camera.CastRay(x, y, w, h, 0.5, 0.5, rnd)
			out = append(out, sampler.Sample(scene, ray, rnd))
		}
	}
	return out
}

func TestRenderEquivalence(t *testing.T) {
	dir := t.TempDir()
	writeSceneResources(t, dir)

	// --- reference scene built imperatively, the classic pt way ---
	goScene := &Scene{}
	floorMat := DiffuseMaterial(HexColor(0xCCCCBB))
	lightMat := LightMaterial(White, 50)
	goScene.Add(NewPlane(V(0, 0, 0), V(0, 0, 1), floorMat))
	goScene.Add(NewSphere(V(0, 0, 1), 1, floorMat))
	goScene.Add(NewCube(V(-2, -2, 1), V(-1, -1, 2), floorMat))
	goScene.Add(NewCylinder(0.5, 0, 2, floorMat))
	goScene.Add(NewSphere(V(3, 3, 5), 0.7, lightMat))
	goCamera := LookAt(V(3, 2, 2), V(0, 0, 0), V(0, 1, 0), 40)
	goSampler := NewSampler(4, 3)
	want := renderSceneToColors(t, goScene, &goCamera, goSampler)

	// --- same scene expressed as JSON ---
	doc := &SceneDocument{
		Version: SceneSchemaVersion,
		Camera: &CameraDocument{
			Type: "lookAt", Eye: ptrVec(V(3, 2, 2)), Center: ptrVec(V(0, 0, 0)),
			WorldUp: ptrVec(V(0, 1, 0)), Fovy: ptrFloat(40),
		},
		Sampler:    &SamplerDocument{Type: "default", FirstHitSamples: ptrInt(4), MaxBounces: ptrInt(3)},
		Background: &BackgroundDocument{Color: &Black},
		Materials: []MaterialDocument{
			{ID: "floor", Color: ptrColor(HexColor(0xCCCCBB))},
			{ID: "lamp", Color: &White, Emittance: ptrFloat(50)},
		},
		Shapes: []ShapeDocument{
			{Type: "plane", Material: "floor", Point: ptrVec(V(0, 0, 0)), Normal: ptrVec(V(0, 0, 1))},
			{Type: "sphere", Material: "floor", Center: ptrVec(V(0, 0, 1)), Radius: ptrFloat(1)},
			{Type: "cube", Material: "floor", Min: ptrVec(V(-2, -2, 1)), Max: ptrVec(V(-1, -1, 2))},
			{Type: "cylinder", Material: "floor", Radius: ptrFloat(0.5), Z0: ptrFloat(0), Z1: ptrFloat(2)},
			{Type: "sphere", Material: "lamp", Center: ptrVec(V(3, 3, 5)), Radius: ptrFloat(0.7)},
		},
	}
	path := filepath.Join(dir, "equiv.json")
	if err := SaveSceneDocument(path, doc); err != nil {
		t.Fatal(err)
	}
	compiled, err := LoadScene(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := renderSceneToColors(t, compiled.Scene, compiled.Camera, compiled.Sampler)
	if len(got) != len(want) {
		t.Fatalf("pixel count %d != %d", len(got), len(want))
	}
	for i := range want {
		if !colorsClose(want[i], got[i], 1e-12) {
			t.Fatalf("pixel %d differs: Go=%v JSON=%v", i, want[i], got[i])
		}
	}
}

func ptrColor(c Color) *Color { return &c }

func colorsClose(a, b Color, eps float64) bool {
	return math.Abs(a.R-b.R) <= eps && math.Abs(a.G-b.G) <= eps && math.Abs(a.B-b.B) <= eps
}

// ---------------------------------------------------------------------------
// reverse direction: Go scene -> document -> Go scene
// ---------------------------------------------------------------------------

func TestNewDocumentFromSceneRoundTrip(t *testing.T) {
	dir := t.TempDir()
	writeSceneResources(t, dir)
	scene := &Scene{}
	mat := GlossyMaterial(HexColor(0x3366CC), 1.4, Radians(20))
	scene.Add(NewPlane(V(0, 0, 0), V(0, 0, 1), mat))
	scene.Add(NewSphere(V(0, 0, 1), 1, mat))
	scene.Add(NewCube(V(1, 1, 1), V(2, 2, 2), mat))
	scene.Add(NewCylinder(0.3, 0, 1, mat))
	mesh := NewMesh([]*Triangle{NewTriangle(
		V(0, 0, 0), V(1, 0, 0), V(0, 1, 0), V(0, 0, 0), V(0, 0, 0), V(0, 0, 0), mat)})
	scene.Add(mesh)
	camera := LookAt(V(2, 2, 2), V(0, 0, 0), V(0, 1, 0), 50)
	sampler := NewDirectSampler()

	doc, err := NewDocumentFromScene(scene, &camera, sampler, nil)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "from-go.json")
	if err := SaveSceneDocument(path, doc); err != nil {
		t.Fatal(err)
	}
	compiled, err := LoadScene(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.Scene.Shapes) != len(scene.Shapes) {
		t.Fatalf("shape count %d != %d", len(compiled.Scene.Shapes), len(scene.Shapes))
	}
	// Re-serialization must be byte-stable.
	data1, err := MarshalSceneDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := LoadSceneDocument(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	data2, err := MarshalSceneDocument(reloaded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data1, data2) {
		t.Fatal("document rebuilt from Go is not byte-stable")
	}
}

func TestSymlinkEscapeRejected(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("symlink permission checks are ineffective as root")
	}
	dir := t.TempDir()
	outside := t.TempDir()
	writePNGTexture(t, filepath.Join(outside, "secret.png"), 2, 2, func(x, y int) color.Color {
		return color.RGBA{1, 2, 3, 255}
	})
	if err := os.Symlink(filepath.Join(outside, "secret.png"),
		filepath.Join(dir, "link.png")); err != nil {
		t.Skipf("symlinks unsupported: %v", err)
	}
	doc := &SceneDocument{
		Version:  SceneSchemaVersion,
		Textures: []TextureDocument{{ID: "t", Path: "link.png"}},
	}
	path := filepath.Join(dir, "scene.json")
	if err := SaveSceneDocument(path, doc); err != nil {
		t.Fatal(err)
	}
	_, err := LoadScene(path, nil)
	if err == nil || !strings.Contains(err.Error(), "symlink") {
		t.Fatalf("expected symlink escape error, got %v", err)
	}
}

func TestOBJAndSTLLoading(t *testing.T) {
	dir := t.TempDir()
	writeSceneResources(t, dir)
	for _, tc := range []struct {
		name        string
		meshPath    string
		format      string
		wantTriMesh bool
	}{
		{"obj inferred", "tri.obj", "", true},
		{"stl inferred", "tri.stl", "", false},
		{"stl explicit", "tri.stl", "stl", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			doc := &SceneDocument{
				Version:   SceneSchemaVersion,
				Materials: []MaterialDocument{{ID: "m", Color: &White}},
				Meshes:    []MeshDocument{{ID: "mesh", Path: tc.meshPath, Format: tc.format, Material: "m"}},
				Shapes:    []ShapeDocument{{Type: "mesh", Mesh: "mesh"}},
			}
			path := filepath.Join(dir, "scene-"+tc.name+".json")
			if err := SaveSceneDocument(path, doc); err != nil {
				t.Fatal(err)
			}
			compiled, err := LoadScene(path, nil)
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			if len(compiled.Scene.Shapes) != 1 {
				t.Fatalf("shape count = %d", len(compiled.Scene.Shapes))
			}
			mesh, ok := compiled.Scene.Shapes[0].(*Mesh)
			if !ok {
				t.Fatalf("expected *Mesh, got %T", compiled.Scene.Shapes[0])
			}
			if len(mesh.Triangles) != 1 {
				t.Fatalf("triangle count = %d, want 1", len(mesh.Triangles))
			}
		})
	}
}

func TestNestedTransformComposition(t *testing.T) {
	dir := t.TempDir()
	mat := DiffuseMaterial(White)
	// inner scale 2, outer translate by (10, 0, 0): world = T * S
	inner := NewTransformedShape(NewSphere(V(0, 0, 0), 1, mat), Scale(V(2, 2, 2)))
	outer := NewTransformedShape(inner, Translate(V(10, 0, 0)))

	doc, err := NewDocumentFromScene(&Scene{Shapes: []Shape{outer}}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileSceneDocument(doc, &LoadOptions{Root: dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.Scene.Shapes) != 1 {
		t.Fatalf("shape count = %d, want 1", len(compiled.Scene.Shapes))
	}
	ts, ok := compiled.Scene.Shapes[0].(*TransformedShape)
	if !ok {
		t.Fatalf("expected TransformedShape, got %T", compiled.Scene.Shapes[0])
	}
	// The scaled+translated sphere center should land at (10, 0, 0) and
	// its surface at radius 2: a ray along +X from the origin must hit at t=8.
	ray := Ray{Origin: V(0, 0, 0), Direction: V(1, 0, 0)}
	hit := ts.Intersect(ray)
	if !hit.Ok() {
		t.Fatal("ray missed the composed shape")
	}
	if math.Abs(hit.T-8) > 1e-9 {
		t.Fatalf("hit distance = %v, want 8", hit.T)
	}
}

func mustMarshal(t *testing.T, doc *SceneDocument) []byte {
	t.Helper()
	data, err := MarshalSceneDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// Hand-written JSON must tolerate the array forms and reject overflow.
func TestHandWrittenJSON(t *testing.T) {
	dir := t.TempDir()
	writePNGTexture(t, filepath.Join(dir, "env.png"), 2, 2, func(x, y int) color.Color {
		return color.RGBA{10, 20, 30, 255}
	})
	hand := `{
  "version": 1,
  "camera": {"type": "basis", "origin": [0,0,5], "right": [1,0,0],
             "up": [0,1,0], "forward": [0,0,-1], "m": 1.5},
  "background": {"color": [0.1, 0.2, 0.3]},
  "materials": [{"id": "m", "color": [0.5, 0.5, 0.5]}],
  "shapes": [
    {"type": "sphere", "material": "m", "center": [0,0,0], "radius": 1}
  ]
}`
	path := filepath.Join(dir, "hand.json")
	if err := os.WriteFile(path, []byte(hand), 0644); err != nil {
		t.Fatal(err)
	}
	compiled, err := LoadScene(path, nil)
	if err != nil {
		t.Fatalf("hand-written JSON failed: %v", err)
	}
	if got := compiled.Scene.Color; got.G != 0.2 {
		t.Fatalf("array color decode wrong: %v", got)
	}
	// A number that overwrites float64 must be rejected, not silently Inf.
	for _, bad := range []string{
		`{"version":1,"shapes":[{"type":"sphere","material":"m","center":[1e999,0,0],"radius":1}],"materials":[{"id":"m"}]}`,
		`{"version":1,"materials":[{"id":"m","color":[1e999,0,0]}]}`,
	} {
		bp := filepath.Join(dir, "overflow.json")
		if err := os.WriteFile(bp, []byte(bad), 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadScene(bp, nil); err == nil {
			t.Fatal("overflow number must be rejected")
		} else if !strings.Contains(err.Error(), "non-finite") &&
			!strings.Contains(err.Error(), "cannot unmarshal number") {
			t.Fatalf("error should reject the overflow, got %v", err)
		}
	}
	// malformed JSON and trailing garbage are rejected with context.
	if _, err := UnmarshalSceneDocument([]byte(`{"version":1} garbage`)); err == nil {
		t.Fatal("trailing data must be rejected")
	}
	if _, err := UnmarshalSceneDocument([]byte(`{not json`)); err == nil {
		t.Fatal("malformed JSON must be rejected")
	}
}

// Registered-texture ordering must not depend on map iteration.
func TestTextureRegistrationDeterministic(t *testing.T) {
	scene := &Scene{}
	texA := &ColorTexture{Width: 1, Height: 1, Data: []Color{{R: 1}}}
	texB := &ColorTexture{Width: 1, Height: 1, Data: []Color{{G: 1}}}
	registered := map[Texture]string{texA: "assets/a.png", texB: "assets/b.png"}
	var first []byte
	for i := 0; i < 5; i++ {
		doc, err := NewDocumentFromScene(scene, nil, nil, registered)
		if err != nil {
			t.Fatal(err)
		}
		data, err := MarshalSceneDocument(doc)
		if err != nil {
			t.Fatal(err)
		}
		if first == nil {
			first = data
		} else if !bytes.Equal(first, data) {
			t.Fatal("texture registration order is non-deterministic across runs")
		}
	}
	// sorted order: a.png before b.png
	if idxA, idxB := bytes.Index(first, []byte("a.png")), bytes.Index(first, []byte("b.png")); idxA > idxB {
		t.Fatal("textures not emitted in sorted path order")
	}
}

// A scene with no background and no shapes still round-trips safely.
func TestEmptySceneRoundTrip(t *testing.T) {
	doc := &SceneDocument{Version: SceneSchemaVersion}
	data, err := MarshalSceneDocument(doc)
	if err != nil {
		t.Fatal(err)
	}
	back, err := UnmarshalSceneDocument(data)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileSceneDocument(back, &LoadOptions{Root: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.Scene.Shapes) != 0 || compiled.Camera != nil || compiled.Sampler != nil {
		t.Fatal("empty scene should compile to empty components")
	}
}
