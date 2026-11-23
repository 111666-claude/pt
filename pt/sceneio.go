package pt

// Versioned, human-readable scene description for fogleman/pt.
//
// A scene document is a JSON object with a stable field order and stable
// floating-point formatting (Go's strconv 'g' format, used by encoding/json,
// is deterministic across platforms). Unknown fields are silently ignored on
// load, so newer writers can extend the schema without breaking older
// readers. The top-level "version" number is the schema version; see
// SceneSchemaVersion and MigrateSceneDocument.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// SceneSchemaVersion is the schema version produced by this build.
const SceneSchemaVersion = 1

// Default limits guarding against hostile or corrupt documents. They can be
// adjusted per LoadOptions; zero values mean "use the built-in default".
const (
	DefaultMaxDocumentBytes    = 64 * 1024 * 1024
	DefaultMaxMaterials        = 4096
	DefaultMaxTextures         = 4096
	DefaultMaxMeshes           = 4096
	DefaultMaxShapes           = 1 << 20
	DefaultMaxTrianglesPerMesh = 5 * 1000 * 1000
	DefaultMaxInlineTriangles  = 10 * 1000 * 1000
	DefaultMaxReferenceDepth   = 256
	DefaultMaxPathLength       = 4096
	DefaultMaxMatrixBytes      = 4096
)

// LoadOptions controls loading and validation of a scene document.
type LoadOptions struct {
	// Root is the directory that relative texture/mesh paths are resolved
	// against. If empty, the directory of the scene file is used, or the
	// current working directory when loading from a reader.
	Root string
	// Limits; zero means use the corresponding Default* constant.
	MaxDocumentBytes    int64
	MaxMaterials        int
	MaxTextures         int
	MaxMeshes           int
	MaxShapes           int
	MaxTrianglesPerMesh int
	MaxInlineTriangles  int
	MaxReferenceDepth   int
	MaxPathLength       int
}

func (o LoadOptions) limits() LoadOptions {
	out := o
	fill := func(v, def int) int {
		if v == 0 {
			return def
		}
		return v
	}
	fill64 := func(v, def int64) int64 {
		if v == 0 {
			return def
		}
		return v
	}
	out.MaxDocumentBytes = fill64(out.MaxDocumentBytes, DefaultMaxDocumentBytes)
	out.MaxMaterials = fill(out.MaxMaterials, DefaultMaxMaterials)
	out.MaxTextures = fill(out.MaxTextures, DefaultMaxTextures)
	out.MaxMeshes = fill(out.MaxMeshes, DefaultMaxMeshes)
	out.MaxShapes = fill(out.MaxShapes, DefaultMaxShapes)
	out.MaxTrianglesPerMesh = fill(out.MaxTrianglesPerMesh, DefaultMaxTrianglesPerMesh)
	out.MaxInlineTriangles = fill(out.MaxInlineTriangles, DefaultMaxInlineTriangles)
	out.MaxReferenceDepth = fill(out.MaxReferenceDepth, DefaultMaxReferenceDepth)
	out.MaxPathLength = fill(out.MaxPathLength, DefaultMaxPathLength)
	return out
}

// CompiledScene is the fully resolved result of loading a document: a ready
// to render Scene plus the camera and sampler and the root directory used to
// resolve relative resource paths.
type CompiledScene struct {
	Scene    *Scene
	Camera   *Camera
	Sampler  Sampler
	Root     string
	Document *SceneDocument
}

// ---------------------------------------------------------------------------
// Document DTOs. Struct field order is the canonical on-disk field order.
// Do not reorder fields; append new ones instead and bump the schema version.
// ---------------------------------------------------------------------------

// SceneDocument is the root object of a .ptscene.json file.
type SceneDocument struct {
	Version    int                 `json:"version"`
	Camera     *CameraDocument     `json:"camera,omitempty"`
	Sampler    *SamplerDocument    `json:"sampler,omitempty"`
	Background *BackgroundDocument `json:"background,omitempty"`
	Materials  []MaterialDocument  `json:"materials,omitempty"`
	Textures   []TextureDocument   `json:"textures,omitempty"`
	Meshes     []MeshDocument      `json:"meshes,omitempty"`
	Shapes     []ShapeDocument     `json:"shapes,omitempty"`
}

// CameraDocument stores the camera basis directly so round-tripping is exact.
// The convenience LookAt form (eye/center/up/fovy) is accepted on load and
// converted to the basis form.
type CameraDocument struct {
	Type    string         `json:"type"`
	Origin  Vector         `json:"origin"`
	Right   Vector         `json:"right"`
	Up      Vector         `json:"up"`
	Forward Vector         `json:"forward"`
	M       float64        `json:"m"`
	Focus   *FocusDocument `json:"focus,omitempty"`
	// LookAt form (input only; never emitted).
	Eye     *Vector  `json:"eye,omitempty"`
	Center  *Vector  `json:"center,omitempty"`
	WorldUp *Vector  `json:"worldUp,omitempty"`
	Fovy    *float64 `json:"fovy,omitempty"`
}

type FocusDocument struct {
	FocalPoint     Vector  `json:"focalPoint"`
	ApertureRadius float64 `json:"apertureRadius"`
}

// SamplerDocument describes a DefaultSampler.
type SamplerDocument struct {
	Type            string  `json:"type"`
	FirstHitSamples *int    `json:"firstHitSamples,omitempty"`
	MaxBounces      *int    `json:"maxBounces,omitempty"`
	DirectLighting  *bool   `json:"directLighting,omitempty"`
	SoftShadows     *bool   `json:"softShadows,omitempty"`
	LightMode       *string `json:"lightMode,omitempty"`
	SpecularMode    *string `json:"specularMode,omitempty"`
}

// BackgroundDocument configures the environment: either a flat color or an
// equirectangular texture.
type BackgroundDocument struct {
	Color        *Color   `json:"color,omitempty"`
	Texture      string   `json:"texture,omitempty"`
	TextureAngle *float64 `json:"textureAngle,omitempty"`
}

// MaterialDocument mirrors pt.Material. Texture slots reference texture IDs.
type MaterialDocument struct {
	ID             string   `json:"id"`
	Color          *Color   `json:"color,omitempty"`
	Texture        string   `json:"texture,omitempty"`
	NormalTexture  string   `json:"normalTexture,omitempty"`
	BumpTexture    string   `json:"bumpTexture,omitempty"`
	GlossTexture   string   `json:"glossTexture,omitempty"`
	BumpMultiplier *float64 `json:"bumpMultiplier,omitempty"`
	Emittance      *float64 `json:"emittance,omitempty"`
	Index          *float64 `json:"index,omitempty"`
	Gloss          *float64 `json:"gloss,omitempty"`
	Tint           *float64 `json:"tint,omitempty"`
	Reflectivity   *float64 `json:"reflectivity,omitempty"`
	Transparent    *bool    `json:"transparent,omitempty"`
}

// TextureDocument is a referenced image file. Pow/scale are applied after
// decoding and gamma correction, matching NewTexture(...).Pow(...).MulScalar(...).
type TextureDocument struct {
	ID    string   `json:"id"`
	Path  string   `json:"path"`
	Pow   *float64 `json:"pow,omitempty"`
	Scale *float64 `json:"scale,omitempty"`
}

// MeshDocument references an OBJ/STL file or carries inline triangles.
type MeshDocument struct {
	ID            string             `json:"id"`
	Path          string             `json:"path,omitempty"`
	Format        string             `json:"format,omitempty"`   // obj | stl, inferred from extension when empty
	Material      string             `json:"material,omitempty"` // parent material for file meshes
	Triangles     []TriangleDocument `json:"triangles,omitempty"`
	SmoothNormals *bool              `json:"smoothNormals,omitempty"`
}

// TriangleDocument stores three vertices, three normals and three UVs.
type TriangleDocument struct {
	V1       Vector `json:"v1"`
	V2       Vector `json:"v2"`
	V3       Vector `json:"v3"`
	N1       Vector `json:"n1,omitempty"`
	N2       Vector `json:"n2,omitempty"`
	N3       Vector `json:"n3,omitempty"`
	T1       Vector `json:"t1,omitempty"`
	T2       Vector `json:"t2,omitempty"`
	T3       Vector `json:"t3,omitempty"`
	Material string `json:"material,omitempty"`
}

// ShapeDocument is one node in the shape list. Type selects the geometry;
// Ref references another shape ID (wrapped in an optional transform);
// Mesh references a mesh ID.
type ShapeDocument struct {
	ID       string   `json:"id,omitempty"`
	Type     string   `json:"type"`
	Material string   `json:"material,omitempty"`
	Ref      string   `json:"ref,omitempty"`
	Mesh     string   `json:"mesh,omitempty"`
	Matrix   *Matrix4 `json:"matrix,omitempty"`
	// geometry parameters, used depending on Type
	Center *Vector  `json:"center,omitempty"`
	Radius *float64 `json:"radius,omitempty"`
	Point  *Vector  `json:"point,omitempty"`
	Normal *Vector  `json:"normal,omitempty"`
	Min    *Vector  `json:"min,omitempty"`
	Max    *Vector  `json:"max,omitempty"`
	Z0     *float64 `json:"z0,omitempty"`
	Z1     *float64 `json:"z1,omitempty"`
}

// Matrix4 is a 4x4 matrix in row-major order, exactly 16 numbers.
type Matrix4 [16]float64

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// MarshalSceneDocument serializes a document with the project's stable
// formatting: two-space indentation, a trailing newline, fixed field order
// and Go's deterministic shortest-round-trip float encoding.
func MarshalSceneDocument(doc *SceneDocument) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(doc); err != nil {
		return nil, &SchemaError{Path: "$", Err: err}
	}
	return buf.Bytes(), nil
}

// UnmarshalSceneDocument parses JSON without touching the filesystem. Unknown
// fields are ignored. The document is structurally validated but resources
// are not loaded; call CompileSceneDocument for that.
func UnmarshalSceneDocument(data []byte) (*SceneDocument, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	// Default decoder: unknown fields are ignored by design.
	var doc SceneDocument
	if err := dec.Decode(&doc); err != nil {
		return nil, &SchemaError{Path: "$", Err: fmt.Errorf("invalid JSON: %w", err)}
	}
	if err := checkNoTrailingData(dec); err != nil {
		return nil, err
	}
	if err := doc.validate(); err != nil {
		return nil, err
	}
	return &doc, nil
}

func checkNoTrailingData(dec *json.Decoder) error {
	var extra json.RawMessage
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return &SchemaError{Path: "$", Err: fmt.Errorf("unexpected data after document")}
		}
		return &SchemaError{Path: "$", Err: err}
	}
	return nil
}

// SaveSceneDocument validates and atomically writes the document to path.
// The file is written to a sibling temp file and renamed, so a failure never
// leaves a truncated scene file behind.
func SaveSceneDocument(path string, doc *SceneDocument) error {
	if doc.Version == 0 {
		doc.Version = SceneSchemaVersion
	}
	if err := doc.validate(); err != nil {
		return err
	}
	data, err := MarshalSceneDocument(doc)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".ptscene-*.tmp")
	if err != nil {
		return &SchemaError{Path: "$", Err: fmt.Errorf("cannot create temp file: %w", err)}
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return &SchemaError{Path: "$", Err: err}
	}
	if err := tmp.Close(); err != nil {
		return &SchemaError{Path: "$", Err: err}
	}
	if err := os.Rename(tmpName, path); err != nil {
		return &SchemaError{Path: "$", Err: err}
	}
	return nil
}

// LoadSceneDocument reads and validates a document from path. No referenced
// resources are opened; use CompileSceneDocument or LoadScene for that.
// opts may be nil for built-in limits.
func LoadSceneDocument(path string, opts *LoadOptions) (*SceneDocument, error) {
	limits := LoadOptions{}.limits()
	if opts != nil {
		limits = opts.limits()
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil, &SchemaError{Path: "$", Err: fmt.Errorf("scene file: %w", err)}
	}
	if info.IsDir() {
		return nil, &SchemaError{Path: "$", Err: fmt.Errorf("scene file is a directory: %s", path)}
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, &SchemaError{Path: "$", Err: fmt.Errorf("scene file: %w", err)}
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limits.MaxDocumentBytes+1))
	if err != nil {
		return nil, &SchemaError{Path: "$", Err: err}
	}
	if int64(len(data)) > limits.MaxDocumentBytes {
		return nil, &SchemaError{
			Path: "$",
			Err:  fmt.Errorf("document exceeds maximum size of %d bytes", limits.MaxDocumentBytes)}
	}
	doc, err := UnmarshalSceneDocument(data)
	if err != nil {
		return nil, err
	}
	if err := doc.validateWithLimits(limits); err != nil {
		return nil, err
	}
	return doc, nil
}

// LoadScene reads a document and resolves it into a renderable scene.
// Relative resource paths are resolved against the scene file's directory.
// On failure the returned *CompiledScene is nil and the error carries the
// JSON path of the offending element. No partially built scene is returned.
func LoadScene(path string, opts *LoadOptions) (*CompiledScene, error) {
	doc, err := LoadSceneDocument(path, opts)
	if err != nil {
		return nil, err
	}
	o := LoadOptions{}
	if opts != nil {
		o = *opts
	}
	if o.Root == "" {
		if abs, err := filepath.Abs(path); err == nil {
			o.Root = filepath.Dir(abs)
		} else {
			o.Root = filepath.Dir(path)
		}
	}
	return CompileSceneDocument(doc, &o)
}

// SchemaError wraps a validation or resolution error with the JSON path of
// the element that caused it (e.g. "shapes[3].matrix").
type SchemaError struct {
	Path string
	Err  error
}

func (e *SchemaError) Error() string {
	if e.Path == "" || e.Path == "$" {
		return e.Err.Error()
	}
	return fmt.Sprintf("%s: %s", e.Path, e.Err)
}

func (e *SchemaError) Unwrap() error { return e.Err }

func schemaError(path string, format string, args ...interface{}) error {
	return &SchemaError{Path: path, Err: fmt.Errorf(format, args...)}
}
