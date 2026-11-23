package pt

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// Compilation of a validated document into live objects. All construction
// happens on local state: nothing is attached to a Scene until every element
// has been built successfully, so a load failure cannot pollute any existing
// scene (or the global texture cache).

// filepathIsVolumeAbs reports whether name looks like a Windows drive-absolute
// path (e.g. C:\...). The JSON paths themselves use slash semantics.
func filepathIsVolumeAbs(name string) bool {
	if len(name) < 3 {
		return false
	}
	c := name[0]
	return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) && name[1] == ':' &&
		(name[2] == '/' || name[2] == '\\')
}

type sceneLoader struct {
	doc        *SceneDocument
	opts       LoadOptions
	root       string
	textures   map[string]Texture
	materials  map[string]Material
	meshes     map[string]*Mesh
	shapes     map[string]Shape
	shapeOrder []Shape
	texCache   map[string]Texture // by resolved absolute path
	buildOne   func(i int) (Shape, error)
}

// CompileSceneDocument resolves a validated document into a renderable
// scene. Resource paths are relative to opts.Root (or the working directory
// when empty). The build is atomic: on error no Scene is returned and the
// global texture cache is untouched.
func CompileSceneDocument(doc *SceneDocument, opts *LoadOptions) (*CompiledScene, error) {
	if err := doc.validate(); err != nil {
		return nil, err
	}
	o := LoadOptions{}.limits()
	if opts != nil {
		given := opts.limits()
		given.Root = opts.Root
		o = given
	}
	root := o.Root
	if root == "" {
		if cwd, err := os.Getwd(); err == nil {
			root = cwd
		}
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	l := &sceneLoader{
		doc:       doc,
		opts:      o,
		root:      root,
		textures:  make(map[string]Texture),
		materials: make(map[string]Material),
		meshes:    make(map[string]*Mesh),
		shapes:    make(map[string]Shape),
		texCache:  make(map[string]Texture),
	}
	if err := l.buildTextures(); err != nil {
		return nil, err
	}
	if err := l.buildMaterials(); err != nil {
		return nil, err
	}
	if err := l.buildMeshes(); err != nil {
		return nil, err
	}
	if err := l.buildShapes(); err != nil {
		return nil, err
	}

	// Everything resolved successfully: only now do we touch the Scene.
	scene := &Scene{}
	if doc.Background != nil {
		if doc.Background.Texture != "" {
			t, ok := l.textures[doc.Background.Texture]
			if !ok {
				return nil, schemaError("$.background.texture",
					"references unknown texture id %q", doc.Background.Texture)
			}
			scene.Texture = t
			if doc.Background.TextureAngle != nil {
				scene.TextureAngle = *doc.Background.TextureAngle
			}
		} else if doc.Background.Color != nil {
			scene.Color = *doc.Background.Color
		}
	}
	for _, shape := range l.shapeOrder {
		scene.Add(shape)
	}

	var camera *Camera
	if doc.Camera != nil {
		c, err := l.buildCamera()
		if err != nil {
			return nil, err
		}
		camera = c
	}
	sampler, err := l.buildSampler()
	if err != nil {
		return nil, err
	}
	return &CompiledScene{
		Scene:    scene,
		Camera:   camera,
		Sampler:  sampler,
		Root:     root,
		Document: doc,
	}, nil
}

// resolvePath maps a slash-separated relative resource path to an OS path
// under the root and re-checks the containment rule after cleaning and after
// resolving symlinks, so a link placed inside the root cannot point outside.
func (l *sceneLoader) resolvePath(p, name string) (string, error) {
	if err := validateResourcePath(p, name); err != nil {
		return "", err
	}
	clean := path.Clean(name)
	rel := filepath.FromSlash(clean)
	joined := filepath.Join(l.root, rel)
	abs, err := filepath.Abs(joined)
	if err != nil {
		return "", schemaError(p, "cannot resolve path %q: %w", name, err)
	}
	rootWithSep := l.root + string(os.PathSeparator)
	if abs != l.root && !strings.HasPrefix(abs, rootWithSep) {
		return "", schemaError(p, "path escapes the scene root: %q", name)
	}
	// Symlink containment: evaluate the real path of the deepest existing
	// ancestor and verify it stays within the real root. A dangling link
	// (target does not exist yet) falls through; the subsequent open reports
	// the missing file with the same contextual path.
	realRoot, err := filepath.EvalSymlinks(l.root)
	if err != nil {
		return "", schemaError(p, "cannot resolve scene root: %w", err)
	}
	existing := abs
	for {
		if _, err := os.Stat(existing); err == nil {
			break
		}
		parent := filepath.Dir(existing)
		if parent == existing {
			break
		}
		existing = parent
	}
	if resolved, err := filepath.EvalSymlinks(existing); err == nil {
		realRootWithSep := realRoot + string(os.PathSeparator)
		if resolved != realRoot && !strings.HasPrefix(resolved, realRootWithSep) {
			return "", schemaError(p, "path escapes the scene root via a symlink: %q", name)
		}
	}
	return abs, nil
}

func (l *sceneLoader) loadTextureFile(p, name string, pow, scale *float64) (Texture, error) {
	abs, err := l.resolvePath(p, name)
	if err != nil {
		return nil, err
	}
	cacheKey := abs
	if t, ok := l.texCache[cacheKey]; ok {
		return t, nil
	}
	t, err := LoadTexture(abs)
	if err != nil {
		return nil, schemaError(p, "cannot load texture %q: %w", name, err)
	}
	if pow != nil {
		t = t.Pow(*pow)
	}
	if scale != nil {
		t = t.MulScalar(*scale)
	}
	l.texCache[cacheKey] = t
	return t, nil
}

func (l *sceneLoader) buildTextures() error {
	for i := range l.doc.Textures {
		t := &l.doc.Textures[i]
		p := fmt.Sprintf("$.textures[%d]", i)
		tex, err := l.loadTextureFile(p+".path", t.Path, t.Pow, t.Scale)
		if err != nil {
			return err
		}
		l.textures[t.ID] = tex
	}
	return nil
}

func (l *sceneLoader) buildMaterials() error {
	for i := range l.doc.Materials {
		d := &l.doc.Materials[i]
		p := fmt.Sprintf("$.materials[%d]", i)
		m := defaultMaterial()
		if d.Color != nil {
			m.Color = *d.Color
		}
		if d.BumpMultiplier != nil {
			m.BumpMultiplier = *d.BumpMultiplier
		}
		if d.Emittance != nil {
			m.Emittance = *d.Emittance
		}
		if d.Index != nil {
			m.Index = *d.Index
		}
		if d.Gloss != nil {
			m.Gloss = *d.Gloss
		}
		if d.Tint != nil {
			m.Tint = *d.Tint
		}
		if d.Reflectivity != nil {
			m.Reflectivity = *d.Reflectivity
		}
		if d.Transparent != nil {
			m.Transparent = *d.Transparent
		}
		for _, ref := range []struct {
			id, field string
			slot      *Texture
		}{
			{d.Texture, "texture", &m.Texture},
			{d.NormalTexture, "normalTexture", &m.NormalTexture},
			{d.BumpTexture, "bumpTexture", &m.BumpTexture},
			{d.GlossTexture, "glossTexture", &m.GlossTexture},
		} {
			if ref.id != "" {
				tex, ok := l.textures[ref.id]
				if !ok {
					return schemaError(p+"."+ref.field, "references unknown texture id %q", ref.id)
				}
				*ref.slot = tex
			}
		}
		l.materials[d.ID] = m
	}
	return nil
}

func (l *sceneLoader) buildMeshes() error {
	for i := range l.doc.Meshes {
		d := &l.doc.Meshes[i]
		p := fmt.Sprintf("$.meshes[%d]", i)
		var parent Material
		if d.Material != "" {
			parent = l.materials[d.Material]
		}
		var mesh *Mesh
		if d.Path != "" {
			built, err := l.loadMeshFile(p, d, parent)
			if err != nil {
				return err
			}
			mesh = built
		} else {
			built, err := l.buildInlineMesh(p, d, parent)
			if err != nil {
				return err
			}
			mesh = built
		}
		if d.SmoothNormals != nil && *d.SmoothNormals {
			mesh.SmoothNormals()
		}
		l.meshes[d.ID] = mesh
	}
	return nil
}

func (l *sceneLoader) loadMeshFile(p string, d *MeshDocument, parent Material) (*Mesh, error) {
	format := strings.ToLower(d.Format)
	if format == "" {
		switch strings.ToLower(path.Ext(d.Path)) {
		case ".obj":
			format = "obj"
		case ".stl":
			format = "stl"
		default:
			return nil, schemaError(p+".path",
				"cannot infer mesh format from extension %q; set \"format\" to obj or stl", d.Path)
		}
	}
	abs, err := l.resolvePath(p+".path", d.Path)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return nil, schemaError(p+".path", "cannot load mesh %q: %w", d.Path, err)
	}
	if info.IsDir() {
		return nil, schemaError(p+".path", "mesh path is a directory: %q", d.Path)
	}
	switch format {
	case "obj":
		mesh, err := LoadOBJ(abs, parent)
		if err != nil {
			return nil, schemaError(p+".path", "cannot load mesh %q: %w", d.Path, err)
		}
		if len(mesh.Triangles) == 0 {
			return nil, schemaError(p+".path", "mesh %q contains no triangles", d.Path)
		}
		return mesh, nil
	case "stl":
		mesh, err := LoadSTL(abs, parent)
		if err != nil {
			return nil, schemaError(p+".path", "cannot load mesh %q: %w", d.Path, err)
		}
		if len(mesh.Triangles) == 0 {
			return nil, schemaError(p+".path", "mesh %q contains no triangles", d.Path)
		}
		return mesh, nil
	}
	return nil, schemaError(p+".format", "must be obj or stl (got %q)", format)
}

func (l *sceneLoader) buildInlineMesh(p string, d *MeshDocument, fallback Material) (*Mesh, error) {
	if len(d.Triangles) == 0 {
		return nil, schemaError(p, "mesh contains no triangles")
	}
	triangles := make([]*Triangle, 0, len(d.Triangles))
	for j := range d.Triangles {
		td := &d.Triangles[j]
		tp := fmt.Sprintf("%s.triangles[%d]", p, j)
		material := fallback
		if td.Material != "" {
			m, ok := l.materials[td.Material]
			if !ok {
				return nil, schemaError(tp+".material", "references unknown material id %q", td.Material)
			}
			material = m
		}
		t := NewTriangle(td.V1, td.V2, td.V3, td.T1, td.T2, td.T3, material)
		if td.N1 != (Vector{}) {
			t.N1 = td.N1
		}
		if td.N2 != (Vector{}) {
			t.N2 = td.N2
		}
		if td.N3 != (Vector{}) {
			t.N3 = td.N3
		}
		t.FixNormals()
		triangles = append(triangles, t)
	}
	return NewMesh(triangles), nil
}

func (l *sceneLoader) buildShapes() error {
	// Memoized, depth-first construction. Validation already proved the
	// graph has no cycles and bounded its depth, but an active set gives a
	// second line of defense against infinite recursion.
	active := make(map[int]bool)
	var build func(i int) (Shape, error)
	build = func(i int) (Shape, error) {
		d := &l.doc.Shapes[i]
		p := fmt.Sprintf("$.shapes[%d]", i)
		if d.ID != "" {
			if shape, ok := l.shapes[d.ID]; ok {
				return shape, nil
			}
		}
		if active[i] {
			return nil, schemaError(p+".ref", "circular shape reference detected")
		}
		active[i] = true
		defer delete(active, i)

		var shape Shape
		isReference := false
		switch d.Type {
		case "reference":
			isReference = true
			target, ok := l.shapeByID(d.Ref)
			if !ok {
				return nil, schemaError(p+".ref", "references unknown shape id %q", d.Ref)
			}
			shape = target
		case "mesh":
			mesh, ok := l.meshes[d.Mesh]
			if !ok {
				return nil, schemaError(p+".mesh", "references unknown mesh id %q", d.Mesh)
			}
			// Copy so a material override cannot mutate the shared mesh.
			shape = mesh.Copy()
			if d.Material != "" {
				shape.(*Mesh).SetMaterial(l.materials[d.Material])
			}
		case "sphere":
			m, err := l.shapeMaterial(p, d)
			if err != nil {
				return nil, err
			}
			shape = NewSphere(*d.Center, *d.Radius, m)
		case "plane":
			m, err := l.shapeMaterial(p, d)
			if err != nil {
				return nil, err
			}
			shape = NewPlane(*d.Point, *d.Normal, m)
		case "cube":
			m, err := l.shapeMaterial(p, d)
			if err != nil {
				return nil, err
			}
			shape = NewCube(*d.Min, *d.Max, m)
		case "cylinder":
			m, err := l.shapeMaterial(p, d)
			if err != nil {
				return nil, err
			}
			shape = NewCylinder(*d.Radius, *d.Z0, *d.Z1, m)
		default:
			return nil, schemaError(p+".type", "unknown shape type %q", d.Type)
		}
		if d.Matrix != nil {
			shape = NewTransformedShape(shape, d.Matrix.toMatrix())
		}
		if d.ID != "" {
			l.shapes[d.ID] = shape
		}
		// A plain reference re-uses the target's already-added shape; only
		// references carrying their own transform contribute a new node.
		if !(isReference && d.Matrix == nil) {
			l.shapeOrder = append(l.shapeOrder, shape)
		}
		return shape, nil
	}
	l.buildOne = build
	for i := range l.doc.Shapes {
		if _, err := build(i); err != nil {
			return err
		}
	}
	return nil
}

func (l *sceneLoader) shapeByID(id string) (Shape, bool) {
	if s, ok := l.shapes[id]; ok {
		return s, true
	}
	for i := range l.doc.Shapes {
		if l.doc.Shapes[i].ID == id {
			s, err := l.buildOne(i)
			if err != nil {
				return nil, false
			}
			return s, true
		}
	}
	return nil, false
}

func (l *sceneLoader) shapeMaterial(p string, d *ShapeDocument) (Material, error) {
	if d.Material == "" {
		return Material{}, schemaError(p+".material", "shape requires a material id")
	}
	m, ok := l.materials[d.Material]
	if !ok {
		return Material{}, schemaError(p+".material", "references unknown material id %q", d.Material)
	}
	return m, nil
}

func (l *sceneLoader) buildCamera() (*Camera, error) {
	d := l.doc.Camera
	p := "$.camera"
	var camera Camera
	if d.Eye != nil {
		camera = LookAt(*d.Eye, *d.Center, *d.WorldUp, *d.Fovy)
	} else {
		camera = NewBasisCamera(d.Origin, d.Right, d.Up, d.Forward, d.M)
	}
	if d.Focus != nil {
		camera.SetFocus(d.Focus.FocalPoint, d.Focus.ApertureRadius)
	}
	// Guard against degenerate bases that survived editing.
	if !camera.IsFinite() {
		return nil, schemaError(p, "camera contains non-finite values after construction")
	}
	return &camera, nil
}

func (l *sceneLoader) buildSampler() (Sampler, error) {
	d := l.doc.Sampler
	if d == nil {
		return nil, nil
	}
	s := &DefaultSampler{
		FirstHitSamples: 16,
		MaxBounces:      8,
		DirectLighting:  true,
		SoftShadows:     true,
		LightMode:       LightModeRandom,
		SpecularMode:    SpecularModeNaive,
	}
	if d.FirstHitSamples != nil {
		s.FirstHitSamples = *d.FirstHitSamples
	}
	if d.MaxBounces != nil {
		s.MaxBounces = *d.MaxBounces
	}
	if d.DirectLighting != nil {
		s.DirectLighting = *d.DirectLighting
	}
	if d.SoftShadows != nil {
		s.SoftShadows = *d.SoftShadows
	}
	if d.LightMode != nil {
		m, _ := parseLightMode(*d.LightMode)
		s.LightMode = m
	}
	if d.SpecularMode != nil {
		m, _ := parseSpecularMode(*d.SpecularMode)
		s.SpecularMode = m
	}
	return s, nil
}

// NewBasisCamera constructs a camera directly from an orthonormal basis and
// the fovy slope m = 1/tan(fovy/2), enabling exact camera round-tripping.
func NewBasisCamera(origin, right, up, forward Vector, m float64) Camera {
	return Camera{p: origin, u: right, v: up, w: forward, m: m}
}

// CameraComponents returns the stored basis so callers can serialize a
// camera built with LookAt without re-deriving (and rounding) its vectors.
func (c *Camera) Components() (origin, right, up, forward Vector, m float64) {
	return c.p, c.u, c.v, c.w, c.m
}

// FocusComponents returns the focal distance and aperture radius.
func (c *Camera) FocusComponents() (focalDistance, apertureRadius float64) {
	return c.focalDistance, c.apertureRadius
}

// IsFinite reports whether every camera component is a finite number.
func (c *Camera) IsFinite() bool {
	vals := []float64{
		c.p.X, c.p.Y, c.p.Z, c.u.X, c.u.Y, c.u.Z,
		c.v.X, c.v.Y, c.v.Z, c.w.X, c.w.Y, c.w.Z,
		c.m, c.focalDistance, c.apertureRadius,
	}
	for _, x := range vals {
		if !isFinite(x) {
			return false
		}
	}
	return true
}

func defaultMaterial() Material {
	return DiffuseMaterial(Black)
}
