package pt

import "math"

// SDFShape

type SDFShape struct {
	SDF
	Material Material
}

func NewSDFShape(sdf SDF, material Material) Shape {
	return &SDFShape{sdf, material}
}

func (s *SDFShape) Compile() {
}

func (s *SDFShape) Intersect(ray Ray) Hit {
	const epsilon = 0.00001
	const start = 0.0001
	const jumpSize = 0.001
	box := s.BoundingBox()
	t1, t2 := box.Intersect(ray)
	if t2 < t1 || t2 < 0 {
		return NoHit
	}
	t := math.Max(start, t1)
	jump := true
	for i := 0; i < 1000; i++ {
		d := s.Evaluate(ray.Position(t))
		if jump && d < 0 {
			t -= jumpSize
			jump = false
			continue
		}
		if d < epsilon {
			return Hit{s, t, nil}
		}
		if jump && d < jumpSize {
			d = jumpSize
		}
		t += d
		if t > t2 {
			return NoHit
		}
	}
	return NoHit
}

func (s *SDFShape) UV(p Vector) Vector {
	return Vector{}
}

func (s *SDFShape) NormalAt(p Vector) Vector {
	const e = 0.0001
	x, y, z := p.X, p.Y, p.Z
	n := Vector{
		s.Evaluate(Vector{x - e, y, z}) - s.Evaluate(Vector{x + e, y, z}),
		s.Evaluate(Vector{x, y - e, z}) - s.Evaluate(Vector{x, y + e, z}),
		s.Evaluate(Vector{x, y, z - e}) - s.Evaluate(Vector{x, y, z + e}),
	}
	// A zero (or NaN) gradient can occur at degenerate points, e.g. a tangent
	// contact; fall back to a deterministic axis instead of returning NaN.
	if n.Length() < e*e || n.X != n.X {
		return Vector{0, 0, 1}
	}
	return n.Normalize()
}

func (s *SDFShape) MaterialAt(p Vector) Material {
	if m, ok := s.SDF.(SDFMaterial); ok {
		if material, ok := m.MaterialAt(p); ok {
			return material
		}
	}
	return s.Material
}

// SDFMaterial is implemented by SDF nodes that can report the material
// governing a point. A SmoothUnionSDF of differently-materialed children
// delegates through this interface; a node returns ok=false when none of its
// descendants carries an explicit material, in which case the SDFShape's own
// material is used.
type SDFMaterial interface {
	MaterialAt(p Vector) (Material, bool)
}

// MaterialSDF attaches a Material to an SDF subtree so that smooth combination
// nodes can select the correct material at each point.
type MaterialSDF struct {
	SDF
	Mat Material
}

func NewMaterialSDF(sdf SDF, material Material) SDF {
	return &MaterialSDF{sdf, material}
}

func (s *MaterialSDF) MaterialAt(p Vector) (Material, bool) {
	return s.Mat, true
}

// SDF

type SDF interface {
	Evaluate(p Vector) float64
	BoundingBox() Box
}

// sdfEmptyBox is the canonical empty box used by degenerate (item-less)
// combination nodes. It inverts the min/max relationship so that Box.Intersect
// reports every ray as a miss; its components are finite, so it can never
// produce an unbounded bounding box even after transforms and unions.
var sdfEmptyBox = Box{Vector{1, 1, 1}, Vector{-1, -1, -1}}

func sdfIsEmptyBox(box Box) bool {
	return box.Min.X > box.Max.X || box.Min.Y > box.Max.Y || box.Min.Z > box.Max.Z
}

// sdfFiniteBox guarantees that a bounding box has finite components, clamping
// infinities to a large but bounded limit and collapsing NaN components to the
// canonical empty box. This ensures degenerate SDFs (e.g. NaN dimensions) can
// never produce an unbounded bounding box.
func sdfFiniteBox(box Box) Box {
	const limit = 1e8
	if !math.IsNaN(box.Min.X) && !math.IsNaN(box.Min.Y) && !math.IsNaN(box.Min.Z) &&
		!math.IsNaN(box.Max.X) && !math.IsNaN(box.Max.Y) && !math.IsNaN(box.Max.Z) {
		return Box{
			Vector{
				math.Max(-limit, math.Min(limit, box.Min.X)),
				math.Max(-limit, math.Min(limit, box.Min.Y)),
				math.Max(-limit, math.Min(limit, box.Min.Z)),
			},
			Vector{
				math.Max(-limit, math.Min(limit, box.Max.X)),
				math.Max(-limit, math.Min(limit, box.Max.Y)),
				math.Max(-limit, math.Min(limit, box.Max.Z)),
			},
		}
	}
	return sdfEmptyBox
}

// sdfFiniteDistance treats NaN distances as "infinitely far away" so a single
// degenerate child cannot poison a whole combination node.
func sdfFiniteDistance(d float64) float64 {
	if math.IsNaN(d) {
		return math.Inf(1)
	}
	return d
}

// SphereSDF

type SphereSDF struct {
	Radius   float64
	Exponent float64
}

func NewSphereSDF(radius float64) SDF {
	return &SphereSDF{radius, 2}
}

func (s *SphereSDF) Evaluate(p Vector) float64 {
	return sdfFiniteDistance(p.LengthN(s.Exponent) - s.Radius)
}

func (s *SphereSDF) BoundingBox() Box {
	r := s.Radius
	return sdfFiniteBox(Box{Vector{-r, -r, -r}, Vector{r, r, r}})
}

// CubeSDF

type CubeSDF struct {
	Size Vector
}

func NewCubeSDF(size Vector) SDF {
	return &CubeSDF{size}
}

func (s *CubeSDF) Evaluate(p Vector) float64 {
	x := p.X
	y := p.Y
	z := p.Z
	if x < 0 {
		x = -x
	}
	if y < 0 {
		y = -y
	}
	if z < 0 {
		z = -z
	}
	x -= s.Size.X / 2
	y -= s.Size.Y / 2
	z -= s.Size.Z / 2
	a := x
	if y > a {
		a = y
	}
	if z > a {
		a = z
	}
	if a > 0 {
		a = 0
	}
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if z < 0 {
		z = 0
	}
	b := math.Sqrt(x*x + y*y + z*z)
	return sdfFiniteDistance(a + b)
}

func (s *CubeSDF) BoundingBox() Box {
	x, y, z := s.Size.X/2, s.Size.Y/2, s.Size.Z/2
	return sdfFiniteBox(Box{Vector{-x, -y, -z}, Vector{x, y, z}})
}

// CylinderSDF

type CylinderSDF struct {
	Radius float64
	Height float64
}

func NewCylinderSDF(radius, height float64) SDF {
	return &CylinderSDF{radius, height}
}

func (s *CylinderSDF) Evaluate(p Vector) float64 {
	x := math.Sqrt(p.X*p.X + p.Z*p.Z)
	y := p.Y
	if x < 0 {
		x = -x
	}
	if y < 0 {
		y = -y
	}
	x -= s.Radius
	y -= s.Height / 2
	a := x
	if y > a {
		a = y
	}
	if a > 0 {
		a = 0
	}
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	b := math.Sqrt(x*x + y*y)
	return sdfFiniteDistance(a + b)
}

func (s *CylinderSDF) BoundingBox() Box {
	r := s.Radius
	h := s.Height / 2
	return sdfFiniteBox(Box{Vector{-r, -h, -r}, Vector{r, h, r}})
}

// CapsuleSDF

type CapsuleSDF struct {
	A, B     Vector
	Radius   float64
	Exponent float64
}

func NewCapsuleSDF(a, b Vector, radius float64) SDF {
	return &CapsuleSDF{a, b, radius, 2}
}

func (s *CapsuleSDF) Evaluate(p Vector) float64 {
	pa := p.Sub(s.A)
	ba := s.B.Sub(s.A)
	h := math.Max(0, math.Min(1, pa.Dot(ba)/ba.Dot(ba)))
	return sdfFiniteDistance(pa.Sub(ba.MulScalar(h)).LengthN(s.Exponent) - s.Radius)
}

func (s *CapsuleSDF) BoundingBox() Box {
	a, b := s.A.Min(s.B), s.A.Max(s.B)
	return sdfFiniteBox(Box{a.SubScalar(s.Radius), b.AddScalar(s.Radius)})
}

// TorusSDF

type TorusSDF struct {
	MajorRadius   float64
	MinorRadius   float64
	MajorExponent float64
	MinorExponent float64
}

func NewTorusSDF(major, minor float64) SDF {
	return &TorusSDF{major, minor, 2, 2}
}

func (s *TorusSDF) Evaluate(p Vector) float64 {
	q := Vector{Vector{p.X, p.Y, 0}.LengthN(s.MajorExponent) - s.MajorRadius, p.Z, 0}
	return sdfFiniteDistance(q.LengthN(s.MinorExponent) - s.MinorRadius)
}

func (s *TorusSDF) BoundingBox() Box {
	a := s.MinorRadius
	b := s.MinorRadius + s.MajorRadius
	return sdfFiniteBox(Box{Vector{-b, -b, -a}, Vector{b, b, a}})
}

// TransformSDF

type TransformSDF struct {
	SDF
	Matrix  Matrix
	Inverse Matrix
}

func NewTransformSDF(sdf SDF, matrix Matrix) SDF {
	return &TransformSDF{sdf, matrix, matrix.Inverse()}
}

func (s *TransformSDF) Evaluate(p Vector) float64 {
	q := s.Inverse.MulPosition(p)
	return s.SDF.Evaluate(q)
}

func (s *TransformSDF) MaterialAt(p Vector) (Material, bool) {
	if m, ok := s.SDF.(SDFMaterial); ok {
		return m.MaterialAt(s.Inverse.MulPosition(p))
	}
	return Material{}, false
}

func (s *TransformSDF) BoundingBox() Box {
	box := s.SDF.BoundingBox()
	if sdfIsEmptyBox(box) {
		return sdfEmptyBox
	}
	return sdfFiniteBox(s.Matrix.MulBox(box))
}

// ScaleSDF

type ScaleSDF struct {
	SDF
	Factor float64
}

func NewScaleSDF(sdf SDF, factor float64) SDF {
	return &ScaleSDF{sdf, factor}
}

func (s *ScaleSDF) Evaluate(p Vector) float64 {
	return s.SDF.Evaluate(p.DivScalar(s.Factor)) * s.Factor
}

func (s *ScaleSDF) MaterialAt(p Vector) (Material, bool) {
	if m, ok := s.SDF.(SDFMaterial); ok {
		return m.MaterialAt(p.DivScalar(s.Factor))
	}
	return Material{}, false
}

func (s *ScaleSDF) BoundingBox() Box {
	box := s.SDF.BoundingBox()
	if sdfIsEmptyBox(box) {
		return sdfEmptyBox
	}
	f := s.Factor
	m := Scale(Vector{f, f, f})
	return sdfFiniteBox(m.MulBox(box))
}

// UnionSDF

type UnionSDF struct {
	Items []SDF
}

func NewUnionSDF(items ...SDF) SDF {
	return &UnionSDF{items}
}

func (s *UnionSDF) Evaluate(p Vector) float64 {
	if len(s.Items) == 0 {
		return math.Inf(1)
	}
	var result float64
	for i, item := range s.Items {
		d := sdfFiniteDistance(item.Evaluate(p))
		if i == 0 || d < result {
			result = d
		}
	}
	return result
}

func (s *UnionSDF) MaterialAt(p Vector) (Material, bool) {
	index := -1
	var result float64
	for i, item := range s.Items {
		d := sdfFiniteDistance(item.Evaluate(p))
		if i == 0 || d < result {
			result = d
			index = i
		}
	}
	if index < 0 {
		return Material{}, false
	}
	if m, ok := s.Items[index].(SDFMaterial); ok {
		return m.MaterialAt(p)
	}
	return Material{}, false
}

func (s *UnionSDF) BoundingBox() Box {
	if len(s.Items) == 0 {
		return sdfEmptyBox
	}
	result := s.Items[0].BoundingBox()
	for _, item := range s.Items[1:] {
		box := item.BoundingBox()
		if sdfIsEmptyBox(box) {
			continue
		}
		if sdfIsEmptyBox(result) {
			result = box
			continue
		}
		result = result.Extend(box)
	}
	return sdfFiniteBox(result)
}

// DifferenceSDF

type DifferenceSDF struct {
	Items []SDF
}

func NewDifferenceSDF(items ...SDF) SDF {
	return &DifferenceSDF{items}
}

func (s *DifferenceSDF) Evaluate(p Vector) float64 {
	if len(s.Items) == 0 {
		return math.Inf(1)
	}
	result := sdfFiniteDistance(s.Items[0].Evaluate(p))
	for _, item := range s.Items[1:] {
		d := sdfFiniteDistance(item.Evaluate(p))
		if -d > result {
			result = -d
		}
	}
	return result
}

func (s *DifferenceSDF) BoundingBox() Box {
	if len(s.Items) == 0 {
		return sdfEmptyBox
	}
	return sdfFiniteBox(s.Items[0].BoundingBox())
}

// MaterialAt reports the material of the first (base) child, since subtraction
// only carves material away rather than contributing a surface of its own.
func (s *DifferenceSDF) MaterialAt(p Vector) (Material, bool) {
	if len(s.Items) == 0 {
		return Material{}, false
	}
	if m, ok := s.Items[0].(SDFMaterial); ok {
		return m.MaterialAt(p)
	}
	return Material{}, false
}

// IntersectionSDF

type IntersectionSDF struct {
	Items []SDF
}

func NewIntersectionSDF(items ...SDF) SDF {
	return &IntersectionSDF{items}
}

func (s *IntersectionSDF) Evaluate(p Vector) float64 {
	if len(s.Items) == 0 {
		return math.Inf(1)
	}
	var result float64
	for i, item := range s.Items {
		d := sdfFiniteDistance(item.Evaluate(p))
		if i == 0 || d > result {
			result = d
		}
	}
	return result
}

func (s *IntersectionSDF) BoundingBox() Box {
	// TODO: intersect boxes
	if len(s.Items) == 0 {
		return sdfEmptyBox
	}
	result := s.Items[0].BoundingBox()
	for _, item := range s.Items[1:] {
		box := item.BoundingBox()
		if sdfIsEmptyBox(box) {
			continue
		}
		if sdfIsEmptyBox(result) {
			result = box
			continue
		}
		result = result.Extend(box)
	}
	return sdfFiniteBox(result)
}

func (s *IntersectionSDF) MaterialAt(p Vector) (Material, bool) {
	index := -1
	var result float64
	for i, item := range s.Items {
		d := sdfFiniteDistance(item.Evaluate(p))
		if i == 0 || d > result {
			result = d
			index = i
		}
	}
	if index < 0 {
		return Material{}, false
	}
	if m, ok := s.Items[index].(SDFMaterial); ok {
		return m.MaterialAt(p)
	}
	return Material{}, false
}

// SmoothUnionSDF

// SmoothUnionSDF blends its children with a polynomial smooth minimum, so the
// surfaces merge with a continuous, rounded transition instead of a hard seam.
// K controls the blending radius: k <= 0 or a non-finite k falls back to the
// exact hard-union (minimum) semantics of UnionSDF.
type SmoothUnionSDF struct {
	Items []SDF
	K     float64
}

// NewSmoothUnionSDF creates a smooth union with blending radius k. Nested
// smooth unions with the same (finite, positive) k are flattened into this
// node so that arbitrarily deep nesting cannot cause unbounded recursion.
func NewSmoothUnionSDF(k float64, items ...SDF) SDF {
	s := &SmoothUnionSDF{K: k}
	smooth := k > 0 && !math.IsNaN(k) && !math.IsInf(k, 0)
	for _, item := range items {
		if inner, ok := item.(*SmoothUnionSDF); ok && smooth && inner.K == k {
			s.Items = append(s.Items, inner.Items...)
		} else {
			s.Items = append(s.Items, item)
		}
	}
	return s
}

// sdfSmoothMin is the polynomial smooth minimum:
//
//	h = clamp(0.5 + 0.5*(b-a)/k, 0, 1)
//	return lerp(b, a, h) - k*h*(1-h)
//
// It is C1-continuous everywhere, exact when |a-b| >= k, and bounded below by
// min(a, b) - k/4 (the blending bulge reaches k/4 at a == b).
func sdfSmoothMin(a, b, k float64) (result, weight float64) {
	h := 0.5 + 0.5*(b-a)/k
	if h < 0 {
		h = 0
	} else if h > 1 {
		h = 1
	}
	return b*(1-h) + a*h - k*h*(1-h), h
}

// foldAt evaluates the combination at p and returns the blended distance plus
// the index of the item governing the point (used for material selection).
// A k <= 0 or non-finite k takes the exact minimum, matching UnionSDF.
func (s *SmoothUnionSDF) foldAt(p Vector) (float64, int) {
	if len(s.Items) == 0 {
		return math.Inf(1), -1
	}
	if !(s.K > 0) || math.IsNaN(s.K) || math.IsInf(s.K, 0) {
		index := 0
		result := sdfFiniteDistance(s.Items[0].Evaluate(p))
		for i, item := range s.Items[1:] {
			d := sdfFiniteDistance(item.Evaluate(p))
			if d < result {
				result = d
				index = i + 1
			}
		}
		return result, index
	}
	index := 0
	result := sdfFiniteDistance(s.Items[0].Evaluate(p))
	for i, item := range s.Items[1:] {
		d := sdfFiniteDistance(item.Evaluate(p))
		m, h := sdfSmoothMin(result, d, s.K)
		if h < 0.5 {
			index = i + 1
		}
		result = m
	}
	return result, index
}

func (s *SmoothUnionSDF) Evaluate(p Vector) float64 {
	result, _ := s.foldAt(p)
	return result
}

// MaterialAt reports the material of the child governing p. Children without
// an explicit material are skipped in favor of the shape-level material; when
// no child carries a material it returns ok=false.
func (s *SmoothUnionSDF) MaterialAt(p Vector) (Material, bool) {
	_, index := s.foldAt(p)
	if index < 0 {
		return Material{}, false
	}
	if m, ok := s.Items[index].(SDFMaterial); ok {
		return m.MaterialAt(p)
	}
	return Material{}, false
}

func (s *SmoothUnionSDF) BoundingBox() Box {
	if len(s.Items) == 0 {
		return sdfEmptyBox
	}
	var result Box
	found := false
	for _, item := range s.Items {
		box := item.BoundingBox()
		if sdfIsEmptyBox(box) {
			continue
		}
		if s.K > 0 && !math.IsNaN(s.K) && !math.IsInf(s.K, 0) {
			// The smooth-minimum bulges outward by at most k/4.
			e := s.K / 4
			box = Box{box.Min.SubScalar(e), box.Max.AddScalar(e)}
		}
		if !found {
			result = box
			found = true
		} else {
			result = result.Extend(box)
		}
	}
	if !found {
		return sdfEmptyBox
	}
	return sdfFiniteBox(result)
}

// RepeatSDF

type RepeatSDF struct {
	SDF
	Step Vector
}

func NewRepeatSDF(sdf SDF, step Vector) SDF {
	return &RepeatSDF{sdf, step}
}

func (s *RepeatSDF) Evaluate(p Vector) float64 {
	q := p.Mod(s.Step).Sub(s.Step.DivScalar(2))
	return s.SDF.Evaluate(q)
}

func (s *RepeatSDF) BoundingBox() Box {
	// TODO: fix this
	return Box{}
}
