package pt

import (
	"fmt"
	"math"
	"path"
	"strings"
)

// Validation of the document DTOs. Everything here is pure: no filesystem
// access and no object construction, so a rejected document can never leak
// half-built state into a Scene.

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func checkFinite(path string, v Vector) error {
	if !isFinite(v.X) || !isFinite(v.Y) || !isFinite(v.Z) {
		return schemaError(path, "vector contains a non-finite component: %v", v)
	}
	return nil
}

func checkFiniteColor(p string, c Color) error {
	if !isFinite(c.R) || !isFinite(c.G) || !isFinite(c.B) {
		return schemaError(p, "color contains a non-finite component: %v", c)
	}
	return nil
}

func checkFiniteScalar(p string, x float64) error {
	if !isFinite(x) {
		return schemaError(p, "non-finite number")
	}
	return nil
}

// validateMatrix checks finiteness and rejects singular/near-singular
// matrices, since TransformedShape computes an inverse and a degenerate
// matrix would produce NaN rays.
func (m *Matrix4) validate(p string) error {
	for i, x := range m {
		if !isFinite(x) {
			return schemaError(p, "matrix element [%d] is non-finite", i)
		}
	}
	det := m.toMatrix().Determinant()
	if !isFinite(det) || det == 0 {
		return schemaError(p, "matrix is singular (determinant %g), it cannot be inverted", det)
	}
	// A determinant this small makes the inverse numerically meaningless.
	scale := m.frobeniusScale()
	if math.Abs(det) < 1e-12*math.Max(1, scale) {
		return schemaError(p, "matrix is nearly singular (determinant %g)", det)
	}
	return nil
}

func (m *Matrix4) frobeniusScale() float64 {
	sum := 0.0
	for _, x := range m {
		sum += x * x
	}
	return sum / 16
}

func (m *Matrix4) toMatrix() Matrix {
	return Matrix{
		m[0], m[1], m[2], m[3],
		m[4], m[5], m[6], m[7],
		m[8], m[9], m[10], m[11],
		m[12], m[13], m[14], m[15]}
}

func matrixFromMatrix(m Matrix) Matrix4 {
	return Matrix4{
		m.x00, m.x01, m.x02, m.x03,
		m.x10, m.x11, m.x12, m.x13,
		m.x20, m.x21, m.x22, m.x23,
		m.x30, m.x31, m.x32, m.x33}
}

// validateResourcePath enforces the relative-root rule: referenced files
// must be relative paths that stay inside the root. Absolute paths and
// traversal sequences are rejected. Checks use slash semantics, matching
// the JSON convention; OS-specific conversion happens at open time.
func validateResourcePath(p, name string) error {
	if name == "" {
		return schemaError(p, "path is empty")
	}
	if len(name) > DefaultMaxPathLength {
		return schemaError(p, "path exceeds maximum length of %d", DefaultMaxPathLength)
	}
	if path.IsAbs(name) || filepathIsVolumeAbs(name) {
		return schemaError(p, "absolute paths are not allowed (paths are relative to the scene root): %q", name)
	}
	clean := path.Clean(name)
	if clean == ".." || strings.HasPrefix(clean, "../") {
		return schemaError(p, "path escapes the scene root: %q", name)
	}
	if strings.Contains(clean, "\\") {
		return schemaError(p, "backslashes are not allowed in resource paths (use forward slashes): %q", name)
	}
	return nil
}

func (doc *SceneDocument) validate() error {
	return doc.validateWithLimits(LoadOptions{}.limits())
}

func (doc *SceneDocument) validateWithLimits(opts LoadOptions) error {
	if doc == nil {
		return schemaError("$", "document is nil")
	}
	switch doc.Version {
	case 0:
		// Unversioned documents are treated as the initial schema. This is
		// documented behavior so hand-written minimal files keep working.
		doc.Version = SceneSchemaVersion
	case SceneSchemaVersion:
	default:
		if doc.Version < SceneSchemaVersion {
			return schemaError("$.version",
				"schema version %d is older than the oldest supported version (%d); no migration is registered",
				doc.Version, SceneSchemaVersion)
		}
		return schemaError("$.version",
			"schema version %d is not supported by this build (supports up to %d); upgrade the renderer or migrate the file",
			doc.Version, SceneSchemaVersion)
	}
	if len(doc.Materials) > opts.MaxMaterials {
		return schemaError("$.materials", "too many materials: %d > %d", len(doc.Materials), opts.MaxMaterials)
	}
	if len(doc.Textures) > opts.MaxTextures {
		return schemaError("$.textures", "too many textures: %d > %d", len(doc.Textures), opts.MaxTextures)
	}
	if len(doc.Meshes) > opts.MaxMeshes {
		return schemaError("$.meshes", "too many meshes: %d > %d", len(doc.Meshes), opts.MaxMeshes)
	}
	if len(doc.Shapes) > opts.MaxShapes {
		return schemaError("$.shapes", "too many shapes: %d > %d", len(doc.Shapes), opts.MaxShapes)
	}
	if err := doc.validateCamera(); err != nil {
		return err
	}
	if err := doc.validateSampler(); err != nil {
		return err
	}
	if err := doc.validateBackground(); err != nil {
		return err
	}
	matIDs, err := doc.validateMaterials(opts)
	if err != nil {
		return err
	}
	texIDs, err := doc.validateTextures(opts)
	if err != nil {
		return err
	}
	if err := doc.validateMeshes(matIDs, opts); err != nil {
		return err
	}
	if err := doc.validateShapes(matIDs, texIDs, opts); err != nil {
		return err
	}
	return nil
}

func (doc *SceneDocument) validateCamera() error {
	c := doc.Camera
	if c == nil {
		return nil
	}
	p := "$.camera"
	if c.Type != "" && c.Type != "lookAt" && c.Type != "basis" {
		return schemaError(p, "unknown camera type %q", c.Type)
	}
	if c.Eye != nil || c.Center != nil || c.WorldUp != nil || c.Fovy != nil {
		// LookAt input form.
		if c.Eye == nil || c.Center == nil || c.WorldUp == nil || c.Fovy == nil {
			return schemaError(p, "lookAt camera requires eye, center, worldUp and fovy")
		}
		for name, v := range map[string]Vector{"eye": *c.Eye, "center": *c.Center, "worldUp": *c.WorldUp} {
			if err := checkFinite(p+"."+name, v); err != nil {
				return err
			}
		}
		if err := checkFiniteScalar(p+".fovy", *c.Fovy); err != nil {
			return err
		}
		if *c.Fovy <= 0 || *c.Fovy >= 180 {
			return schemaError(p+".fovy", "fovy must be in the open interval (0, 180) degrees")
		}
		if c.Eye.Sub(*c.Center).Length() == 0 {
			return schemaError(p, "eye and center must differ")
		}
		if c.WorldUp.Length() == 0 {
			return schemaError(p+".worldUp", "worldUp must not be the zero vector")
		}
		return nil
	}
	if c.Type == "lookAt" {
		return schemaError(p, "lookAt camera requires eye, center, worldUp and fovy")
	}
	for name, v := range map[string]Vector{"origin": c.Origin, "right": c.Right, "up": c.Up, "forward": c.Forward} {
		if err := checkFinite(p+"."+name, v); err != nil {
			return err
		}
	}
	if c.Right.Length() == 0 || c.Up.Length() == 0 || c.Forward.Length() == 0 {
		return schemaError(p, "right, up and forward must all be non-zero vectors")
	}
	if err := checkFiniteScalar(p+".m", c.M); err != nil {
		return err
	}
	if c.M == 0 {
		return schemaError(p+".m", "m must be non-zero")
	}
	if c.Focus != nil {
		fp := p + ".focus"
		if err := checkFinite(fp+".focalPoint", c.Focus.FocalPoint); err != nil {
			return err
		}
		if err := checkFiniteScalar(fp+".apertureRadius", c.Focus.ApertureRadius); err != nil {
			return err
		}
		if c.Focus.ApertureRadius < 0 {
			return schemaError(fp+".apertureRadius", "aperture radius must be non-negative")
		}
	}
	return nil
}

func (doc *SceneDocument) validateSampler() error {
	s := doc.Sampler
	if s == nil {
		return nil
	}
	p := "$.sampler"
	if s.Type != "" && s.Type != "default" {
		return schemaError(p, "unknown sampler type %q", s.Type)
	}
	if s.FirstHitSamples != nil && *s.FirstHitSamples <= 0 {
		return schemaError(p+".firstHitSamples", "must be positive")
	}
	if s.FirstHitSamples != nil {
		n := int(math.Sqrt(float64(*s.FirstHitSamples)))
		if n*n != *s.FirstHitSamples {
			return schemaError(p+".firstHitSamples",
				"must be a perfect square for stratified sampling (got %d)", *s.FirstHitSamples)
		}
	}
	if s.MaxBounces != nil && *s.MaxBounces < 0 {
		return schemaError(p+".maxBounces", "must be non-negative")
	}
	if s.LightMode != nil {
		if _, ok := parseLightMode(*s.LightMode); !ok {
			return schemaError(p+".lightMode", "must be one of random, all (got %q)", *s.LightMode)
		}
	}
	if s.SpecularMode != nil {
		if _, ok := parseSpecularMode(*s.SpecularMode); !ok {
			return schemaError(p+".specularMode", "must be one of naive, first, all (got %q)", *s.SpecularMode)
		}
	}
	return nil
}

func (doc *SceneDocument) validateBackground() error {
	b := doc.Background
	if b == nil {
		return nil
	}
	p := "$.background"
	if b.Color != nil {
		if err := checkFiniteColor(p+".color", *b.Color); err != nil {
			return err
		}
	}
	if b.Texture != "" {
		if err := validateResourcePath(p+".texture", b.Texture); err != nil {
			return err
		}
	}
	if b.Color == nil && b.Texture == "" {
		return schemaError(p, "background requires a color or texture")
	}
	if b.TextureAngle != nil {
		if err := checkFiniteScalar(p+".textureAngle", *b.TextureAngle); err != nil {
			return err
		}
	}
	return nil
}

func (doc *SceneDocument) validateMaterials(opts LoadOptions) (map[string]bool, error) {
	ids := make(map[string]bool, len(doc.Materials))
	for i := range doc.Materials {
		m := &doc.Materials[i]
		p := fmt.Sprintf("$.materials[%d]", i)
		if m.ID == "" {
			return nil, schemaError(p+".id", "material id is required")
		}
		if ids[m.ID] {
			return nil, schemaError(p+".id", "duplicate material id %q", m.ID)
		}
		ids[m.ID] = true
		if m.Color != nil {
			if err := checkFiniteColor(p+".color", *m.Color); err != nil {
				return nil, err
			}
		}
		for name, x := range map[string]*float64{
			"bumpMultiplier": m.BumpMultiplier, "emittance": m.Emittance,
			"index": m.Index, "gloss": m.Gloss, "tint": m.Tint, "reflectivity": m.Reflectivity,
		} {
			if x != nil {
				if err := checkFiniteScalar(p+"."+name, *x); err != nil {
					return nil, err
				}
			}
		}
		if m.Emittance != nil && *m.Emittance < 0 {
			return nil, schemaError(p+".emittance", "must be non-negative")
		}
		if m.BumpMultiplier != nil && *m.BumpMultiplier < 0 {
			return nil, schemaError(p+".bumpMultiplier", "must be non-negative")
		}
	}
	return ids, nil
}

func (doc *SceneDocument) validateTextures(opts LoadOptions) (map[string]bool, error) {
	ids := make(map[string]bool, len(doc.Textures))
	for i := range doc.Textures {
		t := &doc.Textures[i]
		p := fmt.Sprintf("$.textures[%d]", i)
		if t.ID == "" {
			return nil, schemaError(p+".id", "texture id is required")
		}
		if ids[t.ID] {
			return nil, schemaError(p+".id", "duplicate texture id %q", t.ID)
		}
		ids[t.ID] = true
		if err := validateResourcePath(p+".path", t.Path); err != nil {
			return nil, err
		}
		if t.Pow != nil {
			if err := checkFiniteScalar(p+".pow", *t.Pow); err != nil {
				return nil, err
			}
		}
		if t.Scale != nil {
			if err := checkFiniteScalar(p+".scale", *t.Scale); err != nil {
				return nil, err
			}
		}
	}
	return ids, nil
}

func (doc *SceneDocument) validateMeshes(matIDs map[string]bool, opts LoadOptions) error {
	ids := make(map[string]bool, len(doc.Meshes))
	inlineTotal := 0
	for i := range doc.Meshes {
		m := &doc.Meshes[i]
		p := fmt.Sprintf("$.meshes[%d]", i)
		if m.ID == "" {
			return schemaError(p+".id", "mesh id is required")
		}
		if ids[m.ID] {
			return schemaError(p+".id", "duplicate mesh id %q", m.ID)
		}
		ids[m.ID] = true
		hasPath := m.Path != ""
		hasTris := len(m.Triangles) > 0
		if hasPath == hasTris {
			return schemaError(p, "mesh requires exactly one of path or triangles")
		}
		if m.Material != "" && !matIDs[m.Material] {
			return schemaError(p+".material", "references unknown material id %q", m.Material)
		}
		if hasPath {
			if err := validateResourcePath(p+".path", m.Path); err != nil {
				return err
			}
			if m.Format != "" && m.Format != "obj" && m.Format != "stl" {
				return schemaError(p+".format", "must be obj or stl (got %q)", m.Format)
			}
		}
		if hasTris {
			if len(m.Triangles) > opts.MaxTrianglesPerMesh {
				return schemaError(p+".triangles", "mesh has %d triangles, limit is %d",
					len(m.Triangles), opts.MaxTrianglesPerMesh)
			}
			inlineTotal += len(m.Triangles)
			if inlineTotal > opts.MaxInlineTriangles {
				return schemaError("$.meshes", "inline triangle total %d exceeds limit %d",
					inlineTotal, opts.MaxInlineTriangles)
			}
			for j := range m.Triangles {
				if err := m.Triangles[j].validate(fmt.Sprintf("%s.triangles[%d]", p, j), matIDs); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (t *TriangleDocument) validate(p string, matIDs map[string]bool) error {
	for name, v := range map[string]Vector{
		"v1": t.V1, "v2": t.V2, "v3": t.V3,
		"n1": t.N1, "n2": t.N2, "n3": t.N3,
		"t1": t.T1, "t2": t.T2, "t3": t.T3,
	} {
		if err := checkFinite(p+"."+name, v); err != nil {
			return err
		}
	}
	if t.V1 == t.V2 || t.V2 == t.V3 || t.V1 == t.V3 {
		return schemaError(p, "degenerate triangle: repeated vertices")
	}
	if t.Material != "" && !matIDs[t.Material] {
		return schemaError(p+".material", "references unknown material id %q", t.Material)
	}
	return nil
}

var validShapeTypes = map[string]bool{
	"reference": true, "mesh": true, "sphere": true, "plane": true,
	"cube": true, "cylinder": true,
}

func (doc *SceneDocument) validateShapes(matIDs, texIDs map[string]bool, opts LoadOptions) error {
	ids := make(map[string]bool, len(doc.Shapes))
	for i := range doc.Shapes {
		s := &doc.Shapes[i]
		p := fmt.Sprintf("$.shapes[%d]", i)
		if s.ID != "" {
			if ids[s.ID] {
				return schemaError(p+".id", "duplicate shape id %q", s.ID)
			}
			ids[s.ID] = true
		}
		if !validShapeTypes[s.Type] {
			return schemaError(p+".type", "unknown shape type %q", s.Type)
		}
		if s.Material != "" && !matIDs[s.Material] {
			return schemaError(p+".material", "references unknown material id %q", s.Material)
		}
		if s.Matrix != nil {
			if err := s.Matrix.validate(p + ".matrix"); err != nil {
				return err
			}
		}
		if err := s.validateGeometry(p); err != nil {
			return err
		}
	}
	// Reference integrity and cycle detection are whole-graph checks.
	for i := range doc.Shapes {
		s := &doc.Shapes[i]
		p := fmt.Sprintf("$.shapes[%d]", i)
		if s.Type == "reference" {
			if s.Ref == "" {
				return schemaError(p+".ref", "reference shape requires a ref")
			}
			if !ids[s.Ref] {
				return schemaError(p+".ref", "references unknown shape id %q", s.Ref)
			}
		}
		if s.Type == "mesh" {
			if s.Mesh == "" {
				return schemaError(p+".mesh", "mesh shape requires a mesh")
			}
		}
	}
	indexByID := make(map[string]int, len(doc.Shapes))
	for i := range doc.Shapes {
		if doc.Shapes[i].ID != "" {
			indexByID[doc.Shapes[i].ID] = i
		}
	}
	for i := range doc.Shapes {
		if doc.Shapes[i].Type != "reference" {
			continue
		}
		seen := map[int]bool{i: true}
		cur := i
		depth := 0
		for doc.Shapes[cur].Type == "reference" {
			depth++
			if depth > opts.MaxReferenceDepth {
				return schemaError(fmt.Sprintf("$.shapes[%d].ref", i),
					"reference chain exceeds maximum depth of %d", opts.MaxReferenceDepth)
			}
			next, ok := indexByID[doc.Shapes[cur].Ref]
			if !ok {
				break // already reported above
			}
			if seen[next] {
				return schemaError(fmt.Sprintf("$.shapes[%d].ref", i),
					"circular shape reference detected through id %q", doc.Shapes[cur].Ref)
			}
			seen[next] = true
			cur = next
		}
	}
	return nil
}

func (s *ShapeDocument) validateGeometry(p string) error {
	switch s.Type {
	case "reference":
		return nil
	case "mesh":
		if s.Mesh == "" {
			return schemaError(p+".mesh", "mesh shape requires a mesh id")
		}
	case "sphere":
		if s.Center == nil {
			return schemaError(p, "sphere requires center")
		}
		if err := checkFinite(p+".center", *s.Center); err != nil {
			return err
		}
		if s.Radius == nil {
			return schemaError(p, "sphere requires radius")
		}
		if err := checkFiniteScalar(p+".radius", *s.Radius); err != nil {
			return err
		}
		if *s.Radius <= 0 {
			return schemaError(p+".radius", "must be positive")
		}
	case "plane":
		if s.Point == nil {
			return schemaError(p, "plane requires point")
		}
		if err := checkFinite(p+".point", *s.Point); err != nil {
			return err
		}
		if s.Normal == nil {
			return schemaError(p, "plane requires normal")
		}
		if err := checkFinite(p+".normal", *s.Normal); err != nil {
			return err
		}
		if s.Normal.Length() == 0 {
			return schemaError(p+".normal", "must not be the zero vector")
		}
	case "cube":
		if s.Min == nil || s.Max == nil {
			return schemaError(p, "cube requires min and max")
		}
		if err := checkFinite(p+".min", *s.Min); err != nil {
			return err
		}
		if err := checkFinite(p+".max", *s.Max); err != nil {
			return err
		}
		if s.Max.X <= s.Min.X || s.Max.Y <= s.Min.Y || s.Max.Z <= s.Min.Z {
			return schemaError(p, "cube max must be strictly greater than min on every axis")
		}
	case "cylinder":
		if s.Radius == nil {
			return schemaError(p, "cylinder requires radius")
		}
		if err := checkFiniteScalar(p+".radius", *s.Radius); err != nil {
			return err
		}
		if *s.Radius <= 0 {
			return schemaError(p+".radius", "must be positive")
		}
		if s.Z0 == nil || s.Z1 == nil {
			return schemaError(p, "cylinder requires z0 and z1")
		}
		if err := checkFiniteScalar(p+".z0", *s.Z0); err != nil {
			return err
		}
		if err := checkFiniteScalar(p+".z1", *s.Z1); err != nil {
			return err
		}
		if *s.Z1 <= *s.Z0 {
			return schemaError(p, "cylinder requires z1 > z0")
		}
	}
	return nil
}

func parseLightMode(s string) (LightMode, bool) {
	switch s {
	case "random":
		return LightModeRandom, true
	case "all":
		return LightModeAll, true
	}
	return 0, false
}

func parseSpecularMode(s string) (SpecularMode, bool) {
	switch s {
	case "naive":
		return SpecularModeNaive, true
	case "first":
		return SpecularModeFirst, true
	case "all":
		return SpecularModeAll, true
	}
	return 0, false
}

func lightModeName(m LightMode) string {
	switch m {
	case LightModeAll:
		return "all"
	default:
		return "random"
	}
}

func specularModeName(m SpecularMode) string {
	switch m {
	case SpecularModeFirst:
		return "first"
	case SpecularModeAll:
		return "all"
	default:
		return "naive"
	}
}
