package pt

import (
	"fmt"
	"math"
	"sync/atomic"
)

// Resource limits for instance graphs. These are package-level variables so
// applications (and tests) can tune them. Compile panics with a descriptive
// error when a graph exceeds them; call Validate to check beforehand.
var (
	// MaxInstanceDepth bounds how deeply Instances may be nested.
	MaxInstanceDepth = 64
	// MaxInstanceCount bounds the number of unique instance nodes in a
	// single instance graph.
	MaxInstanceCount = 1 << 20
)

// minInstanceDeterminant is the smallest acceptable absolute determinant of
// an instance transform. Matrices below this are (numerically) singular and
// cannot be inverted safely.
const minInstanceDeterminant = 1e-12

var instanceCounter uint64

// Instance is a scene-graph node that shares geometry with other instances.
// It references a shared Shape (typically a *Mesh) instead of copying it, so
// compiling a scene with thousands of instances never duplicates vertex data
// or acceleration structures. Each instance stores its own transform, an
// optional material override, and a stable identifier.
//
// With no material override an Instance behaves like a TransformedShape, and
// Instances may be nested (an Instance whose Shape is another Instance) or
// combined with TransformedShape freely.
//
// The referenced geometry is shared, not copied: modifying it (e.g.
// Mesh.Transform) affects every instance of it. Bounding volumes are not
// cached by Instance, so re-running Compile picks such changes up.
type Instance struct {
	ID       string    // stable, unique identifier for this instance
	Shape    Shape     // shared geometry, never copied
	Matrix   Matrix    // shape-space to world-space transform
	Inverse  Matrix    // world-space to shape-space transform
	Material *Material // optional per-instance material override
}

// NewInstance creates an instance of a shared shape. If id is empty, a
// unique stable id is generated. An error is returned if the shape is nil
// or empty, or if the matrix is singular (not invertible).
func NewInstance(id string, shape Shape, matrix Matrix) (*Instance, error) {
	if shape == nil {
		return nil, fmt.Errorf("pt: cannot create instance of nil shape")
	}
	if mesh, ok := shape.(*Mesh); ok && len(mesh.Triangles) == 0 {
		return nil, fmt.Errorf("pt: cannot create instance of empty mesh")
	}
	if err := checkInstanceMatrix(matrix); err != nil {
		return nil, err
	}
	if id == "" {
		id = fmt.Sprintf("instance-%d", atomic.AddUint64(&instanceCounter, 1))
	}
	return &Instance{id, shape, matrix, matrix.Inverse(), nil}, nil
}

// MustInstance is like NewInstance but panics on error.
func MustInstance(id string, shape Shape, matrix Matrix) *Instance {
	instance, err := NewInstance(id, shape, matrix)
	if err != nil {
		panic(err)
	}
	return instance
}

func checkInstanceMatrix(m Matrix) error {
	d := m.Determinant()
	if math.IsNaN(d) || math.IsInf(d, 0) {
		return fmt.Errorf("pt: instance matrix is not finite (determinant %v)", d)
	}
	if math.Abs(d) < minInstanceDeterminant {
		return fmt.Errorf("pt: instance matrix is singular (determinant %v)", d)
	}
	return nil
}

// SetMaterial sets a per-instance material override. The shared geometry and
// other instances of it are unaffected. Textures on the override are sampled
// in the shared shape's UV space.
func (s *Instance) SetMaterial(material Material) {
	s.Material = &material
}

// ClearMaterial removes any per-instance material override.
func (s *Instance) ClearMaterial() {
	s.Material = nil
}

// Validate checks the instance graph rooted at this node for nil or empty
// shapes, singular matrices, reference cycles, and violations of the
// MaxInstanceDepth / MaxInstanceCount resource limits. It compiles nothing.
func (s *Instance) Validate() error {
	count := 0
	_, err := s.collect(&count, make(map[*Instance]bool), make(map[*Instance]bool), make(map[Shape]bool), 0, nil)
	return err
}

// Compile validates the instance graph and compiles each unique referenced
// shape exactly once. Shared geometry is compiled in place, never copied.
// It panics if the graph is invalid; use Validate to check without panicking.
func (s *Instance) Compile() {
	count := 0
	shapes, err := s.collect(&count, make(map[*Instance]bool), make(map[*Instance]bool), make(map[Shape]bool), 0, nil)
	if err != nil {
		panic(err)
	}
	for _, shape := range shapes {
		shape.Compile()
	}
}

// collect walks the instance graph depth-first, validating as it goes, and
// returns the unique non-instance shapes referenced, in deterministic order.
// stack holds the instances on the current path (cycle detection); done holds
// instances already fully walked; seen holds leaf shapes already collected.
func (s *Instance) collect(count *int, stack, done map[*Instance]bool, seen map[Shape]bool, depth int, out []Shape) ([]Shape, error) {
	if depth > MaxInstanceDepth {
		return nil, fmt.Errorf("pt: instance %q exceeds maximum nesting depth (%d)", s.ID, MaxInstanceDepth)
	}
	*count++
	if *count > MaxInstanceCount {
		return nil, fmt.Errorf("pt: instance %q exceeds maximum instance graph size (%d)", s.ID, MaxInstanceCount)
	}
	if stack[s] {
		return nil, fmt.Errorf("pt: instance %q contains a reference cycle", s.ID)
	}
	if done[s] {
		return out, nil
	}
	if s.Shape == nil {
		return nil, fmt.Errorf("pt: instance %q has a nil shape", s.ID)
	}
	if err := checkInstanceMatrix(s.Matrix); err != nil {
		return nil, fmt.Errorf("pt: instance %q: %v", s.ID, err)
	}
	stack[s] = true
	switch shape := s.Shape.(type) {
	case *Instance:
		var err error
		out, err = shape.collect(count, stack, done, seen, depth+1, out)
		if err != nil {
			return nil, err
		}
	case *Mesh:
		if len(shape.Triangles) == 0 {
			return nil, fmt.Errorf("pt: instance %q references an empty mesh", s.ID)
		}
		if !seen[shape] {
			seen[shape] = true
			out = append(out, shape)
		}
	default:
		if !seen[s.Shape] {
			seen[s.Shape] = true
			out = append(out, s.Shape)
		}
	}
	delete(stack, s)
	done[s] = true
	return out, nil
}

// BoundingBox returns the shared shape's bounding box transformed into world
// space. It is recomputed on every call, so changes to the shared geometry
// are reflected immediately.
func (s *Instance) BoundingBox() Box {
	return s.Matrix.MulBox(s.Shape.BoundingBox())
}

// Intersect transforms the ray into the shared shape's space, intersects the
// shared geometry, and maps the result back to world space. Nested instances
// and TransformedShapes are handled by reusing the inner hit's HitInfo, which
// is already expressed in this instance's shape space.
func (s *Instance) Intersect(r Ray) Hit {
	shapeRay := s.Inverse.MulRay(r)
	hit := s.Shape.Intersect(shapeRay)
	if !hit.Ok() {
		return hit
	}
	var shape Shape
	var shapePosition, shapeNormal Vector
	var material Material
	var inside bool
	if info := hit.HitInfo; info != nil {
		// inner Instance or TransformedShape already computed position,
		// normal, material and inside-ness in our shape space
		shape = info.Shape
		shapePosition = info.Position
		shapeNormal = info.Normal
		material = info.Material
		inside = info.Inside
	} else {
		shape = hit.Shape
		shapePosition = shapeRay.Position(hit.T)
		shapeNormal = shape.NormalAt(shapePosition)
		material = MaterialAt(shape, shapePosition)
		if shapeNormal.Dot(shapeRay.Direction) > 0 {
			shapeNormal = shapeNormal.Negate()
			inside = true
		}
	}
	if s.Material != nil {
		material = *s.Material
	}
	position := s.Matrix.MulPosition(shapePosition)
	normal := s.Inverse.Transpose().MulDirection(shapeNormal)
	ray := Ray{position, normal}
	info := HitInfo{shape, position, normal, ray, material, inside}
	// report the instance itself so identity (and light visibility checks
	// against scene.Lights) work per-instance
	hit.Shape = s
	hit.T = position.Sub(r.Origin).Length()
	hit.HitInfo = &info
	return hit
}

// UV maps a world-space point into the shared shape's space and returns its
// texture coordinates.
func (s *Instance) UV(p Vector) Vector {
	return s.Shape.UV(s.Inverse.MulPosition(p))
}

// NormalAt returns the world-space normal at a world-space point.
func (s *Instance) NormalAt(p Vector) Vector {
	n := s.Shape.NormalAt(s.Inverse.MulPosition(p))
	return s.Inverse.Transpose().MulDirection(n)
}

// MaterialAt returns the material at a world-space point: the per-instance
// override if set, otherwise the shared shape's material.
func (s *Instance) MaterialAt(p Vector) Material {
	if s.Material != nil {
		return *s.Material
	}
	return s.Shape.MaterialAt(s.Inverse.MulPosition(p))
}
