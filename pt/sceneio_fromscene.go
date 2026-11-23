package pt

import (
	"fmt"
	"sort"
)

// Conversion of Go-constructed scenes back into a versioned document, so
// programs built in the classic imperative style gain save/reload for free.
// Only the concrete, serializable shape types are supported; anything else
// (procedural SDFs, volumes, function surfaces) is reported with its type
// name so the caller knows what could not be represented.

// SceneWriter accumulates the DTO entries for a document. Texture and mesh
// resources must be registered with the relative paths they were loaded
// from, since the in-memory objects do not retain their origin.
type SceneWriter struct {
	doc        *SceneDocument
	materials  map[*Material]string
	matVals    map[Material]string
	textures   map[Texture]string
	meshes     map[*Mesh]string
	matCount   int
	texCount   int
	meshCount  int
	shapeCount int
}

// NewSceneWriter starts a document for the given schema version.
func NewSceneWriter() *SceneWriter {
	return &SceneWriter{
		doc:       &SceneDocument{Version: SceneSchemaVersion},
		materials: make(map[*Material]string),
		matVals:   make(map[Material]string),
		textures:  make(map[Texture]string),
		meshes:    make(map[*Mesh]string),
	}
}

// RegisterTexture records that tex was loaded from the given relative path
// and makes it referenceable by an id. Pow/scale describe post-load
// adjustments the caller applied (pass nil when none).
func (w *SceneWriter) RegisterTexture(id, relPath string, tex Texture, pow, scale *float64) error {
	if err := validateResourcePath("texture["+id+"].path", relPath); err != nil {
		return err
	}
	if _, exists := w.textures[tex]; exists {
		return fmt.Errorf("texture already registered with id %q", w.textures[tex])
	}
	for _, t := range w.doc.Textures {
		if t.ID == id {
			return fmt.Errorf("duplicate texture id %q", id)
		}
	}
	w.textures[tex] = id
	w.doc.Textures = append(w.doc.Textures, TextureDocument{
		ID: id, Path: relPath, Pow: pow, Scale: scale,
	})
	return nil
}

// RegisterMesh records that mesh was loaded from relPath (empty for inline
// meshes, which are serialized as triangles).
func (w *SceneWriter) RegisterMesh(id, relPath, materialID string, mesh *Mesh, smoothNormals *bool) error {
	if relPath != "" {
		if err := validateResourcePath("mesh["+id+"].path", relPath); err != nil {
			return err
		}
	}
	for _, m := range w.doc.Meshes {
		if m.ID == id {
			return fmt.Errorf("duplicate mesh id %q", id)
		}
	}
	w.meshes[mesh] = id
	w.doc.Meshes = append(w.doc.Meshes, MeshDocument{
		ID: id, Path: relPath, Material: materialID,
		SmoothNormals: smoothNormals,
	})
	return nil
}

// SetBackground records the environment color or texture and rotation angle.
func (w *SceneWriter) SetBackground(color Color, texture Texture, angle float64) error {
	bg := &BackgroundDocument{}
	if texture != nil {
		id, ok := w.textures[texture]
		if !ok {
			return fmt.Errorf("background texture is not registered; call RegisterTexture first")
		}
		bg.Texture = id
		if angle != 0 {
			a := angle
			bg.TextureAngle = &a
		}
	} else {
		c := color
		bg.Color = &c
	}
	w.doc.Background = bg
	return nil
}

// SetCamera serializes a camera built with LookAt or NewBasisCamera.
func (w *SceneWriter) SetCamera(camera *Camera) {
	origin, right, up, forward, m := camera.Components()
	focal, aperture := camera.FocusComponents()
	cd := &CameraDocument{
		Type:    "basis",
		Origin:  origin,
		Right:   right,
		Up:      up,
		Forward: forward,
		M:       m,
	}
	if aperture > 0 || focal != 0 {
		cd.Focus = &FocusDocument{
			FocalPoint:     origin.Add(forward.MulScalar(focal)),
			ApertureRadius: aperture,
		}
	}
	w.doc.Camera = cd
}

// SetSampler serializes a DefaultSampler.
func (w *SceneWriter) SetSampler(s *DefaultSampler) {
	first := s.FirstHitSamples
	bounces := s.MaxBounces
	direct := s.DirectLighting
	soft := s.SoftShadows
	light := lightModeName(s.LightMode)
	spec := specularModeName(s.SpecularMode)
	w.doc.Sampler = &SamplerDocument{
		Type:            "default",
		FirstHitSamples: &first,
		MaxBounces:      &bounces,
		DirectLighting:  &direct,
		SoftShadows:     &soft,
		LightMode:       &light,
		SpecularMode:    &spec,
	}
}

// AddShape walks a serializable shape, registering any materials/meshes it
// encounters. The optional id lets other shapes reference it.
func (w *SceneWriter) AddShape(id string, shape Shape) error {
	return w.addShape(id, shape, nil)
}

func (w *SceneWriter) addShape(id string, shape Shape, matrix *Matrix4) error {
	switch s := shape.(type) {
	case *Sphere:
		matID, err := w.materialID(&s.Material)
		if err != nil {
			return err
		}
		center, radius := s.Center, s.Radius
		w.doc.Shapes = append(w.doc.Shapes, ShapeDocument{
			ID: id, Type: "sphere", Material: matID, Matrix: matrix,
			Center: &center, Radius: &radius,
		})
	case *Plane:
		matID, err := w.materialID(&s.Material)
		if err != nil {
			return err
		}
		point, normal := s.Point, s.Normal
		w.doc.Shapes = append(w.doc.Shapes, ShapeDocument{
			ID: id, Type: "plane", Material: matID, Matrix: matrix,
			Point: &point, Normal: &normal,
		})
	case *Cube:
		matID, err := w.materialID(&s.Material)
		if err != nil {
			return err
		}
		mn, mx := s.Min, s.Max
		w.doc.Shapes = append(w.doc.Shapes, ShapeDocument{
			ID: id, Type: "cube", Material: matID, Matrix: matrix,
			Min: &mn, Max: &mx,
		})
	case *Cylinder:
		matID, err := w.materialID(&s.Material)
		if err != nil {
			return err
		}
		radius, z0, z1 := s.Radius, s.Z0, s.Z1
		w.doc.Shapes = append(w.doc.Shapes, ShapeDocument{
			ID: id, Type: "cylinder", Material: matID, Matrix: matrix,
			Radius: &radius, Z0: &z0, Z1: &z1,
		})
	case *Mesh:
		meshID, ok := w.meshes[s]
		if !ok {
			// Fall back to inline serialization so no mesh is lost.
			id2, err := w.inlineMesh(s)
			if err != nil {
				return err
			}
			meshID = id2
		}
		matID := ""
		if len(s.Triangles) > 0 && s.Triangles[0].Material != nil {
			m, err := w.materialID(s.Triangles[0].Material)
			if err != nil {
				return err
			}
			matID = m
		}
		w.doc.Shapes = append(w.doc.Shapes, ShapeDocument{
			ID: id, Type: "mesh", Material: matID, Mesh: meshID, Matrix: matrix,
		})
	case *TransformedShape:
		// Compose nested transforms: the outer matrix applies last, matching
		// NewTransformedShape(NewTransformedShape(inner, Mi), Mo), whose
		// effective world matrix is Mo * Mi.
		inner := matrixFromMatrix(s.Matrix)
		combined := inner
		if matrix != nil {
			combined = matrixFromMatrix(matrix.toMatrix().Mul(inner.toMatrix()))
		}
		return w.addShape(id, s.Shape, &combined)
	default:
		return fmt.Errorf("shape type %T cannot be serialized to the scene schema", shape)
	}
	return nil
}

func (w *SceneWriter) inlineMesh(mesh *Mesh) (string, error) {
	w.meshCount++
	id := fmt.Sprintf("mesh%d", w.meshCount)
	tris := make([]TriangleDocument, 0, len(mesh.Triangles))
	for _, t := range mesh.Triangles {
		td := TriangleDocument{
			V1: t.V1, V2: t.V2, V3: t.V3,
			N1: t.N1, N2: t.N2, N3: t.N3,
			T1: t.T1, T2: t.T2, T3: t.T3,
		}
		if t.Material != nil {
			matID, err := w.materialID(t.Material)
			if err != nil {
				return "", err
			}
			td.Material = matID
		}
		tris = append(tris, td)
	}
	w.meshes[mesh] = id
	w.doc.Meshes = append(w.doc.Meshes, MeshDocument{ID: id, Triangles: tris})
	return id, nil
}

// materialID deduplicates materials by pointer first and value second, so
// repeated use of the same literal material still shares one registry entry.
func (w *SceneWriter) materialID(m *Material) (string, error) {
	if id, ok := w.materials[m]; ok {
		return id, nil
	}
	if id, ok := w.matVals[*m]; ok {
		w.materials[m] = id
		return id, nil
	}
	w.matCount++
	id := fmt.Sprintf("material%d", w.matCount)
	if err := w.addMaterial(id, m); err != nil {
		return "", err
	}
	w.materials[m] = id
	w.matVals[*m] = id
	return id, nil
}

func (w *SceneWriter) addMaterial(id string, m *Material) error {
	d := MaterialDocument{ID: id}
	color := m.Color
	d.Color = &color
	for _, slot := range []struct {
		tex   Texture
		field *string
	}{
		{m.Texture, &d.Texture},
		{m.NormalTexture, &d.NormalTexture},
		{m.BumpTexture, &d.BumpTexture},
		{m.GlossTexture, &d.GlossTexture},
	} {
		if slot.tex != nil {
			texID, ok := w.textures[slot.tex]
			if !ok {
				return fmt.Errorf("material %q references an unregistered texture; call RegisterTexture first", id)
			}
			*slot.field = texID
		}
	}
	bump, emittance, index := m.BumpMultiplier, m.Emittance, m.Index
	gloss, tint, reflectivity := m.Gloss, m.Tint, m.Reflectivity
	transparent := m.Transparent
	d.BumpMultiplier, d.Emittance, d.Index = &bump, &emittance, &index
	d.Gloss, d.Tint, d.Reflectivity = &gloss, &tint, &reflectivity
	d.Transparent = &transparent
	w.doc.Materials = append(w.doc.Materials, d)
	return nil
}

// Document returns the assembled document.
func (w *SceneWriter) Document() *SceneDocument {
	return w.doc
}

// Marshal returns the stable JSON encoding of the assembled document.
func (w *SceneWriter) Marshal() ([]byte, error) {
	return MarshalSceneDocument(w.doc)
}

// NewDocumentFromScene builds a document for a scene assembled the classic
// way. Registered textures map live Texture objects to relative paths.
func NewDocumentFromScene(scene *Scene, camera *Camera, sampler Sampler,
	registeredTextures map[Texture]string) (*SceneDocument, error) {
	w := NewSceneWriter()
	// Register in path-sorted order so the emitted texture list is
	// deterministic regardless of map iteration order.
	paths := make([]string, 0, len(registeredTextures))
	pathToTex := make(map[string]Texture, len(registeredTextures))
	for tex, relPath := range registeredTextures {
		if _, seen := pathToTex[relPath]; !seen {
			paths = append(paths, relPath)
		}
		pathToTex[relPath] = tex
	}
	sort.Strings(paths)
	for _, relPath := range paths {
		id := fmt.Sprintf("texture%d", len(w.doc.Textures)+1)
		if err := w.RegisterTexture(id, relPath, pathToTex[relPath], nil, nil); err != nil {
			return nil, err
		}
	}
	if scene.Texture != nil {
		if err := w.SetBackground(Black, scene.Texture, scene.TextureAngle); err != nil {
			return nil, err
		}
	} else {
		if err := w.SetBackground(scene.Color, nil, 0); err != nil {
			return nil, err
		}
	}
	for _, shape := range scene.Shapes {
		w.shapeCount++
		id := fmt.Sprintf("shape%d", w.shapeCount)
		if err := w.AddShape(id, shape); err != nil {
			return nil, err
		}
	}
	if camera != nil {
		w.SetCamera(camera)
	}
	if s, ok := sampler.(*DefaultSampler); ok && s != nil {
		w.SetSampler(s)
	}
	return w.Document(), nil
}

// MigrateSceneDocument is the extension point for schema upgrades. It maps a
// document from an older supported version to the current one. With a single
// released version there is nothing to migrate yet; the function exists so
// the policy (migrate or reject with a reason) is explicit and tested.
func MigrateSceneDocument(doc *SceneDocument) (*SceneDocument, error) {
	if doc == nil {
		return nil, schemaError("$", "document is nil")
	}
	switch doc.Version {
	case 0, SceneSchemaVersion:
		out := *doc
		out.Version = SceneSchemaVersion
		return &out, nil
	default:
		if doc.Version < SceneSchemaVersion {
			return nil, schemaError("$.version",
				"cannot migrate schema version %d: no migration registered", doc.Version)
		}
		return nil, schemaError("$.version",
			"cannot read schema version %d with parser version %d", doc.Version, SceneSchemaVersion)
	}
}
