package pt

import (
	"math"
	"testing"
)

// Independent reference implementations, written separately from the
// production code, used to cross-check the smooth SDF nodes.

// refSphereDistance is the exact signed distance to a sphere centered at c.
func refSphereDistance(p, c Vector, r float64) float64 {
	return p.Sub(c).Length() - r
}

// refSmoothMin is the independently-derived polynomial smooth minimum.
func refSmoothMin(a, b, k float64) float64 {
	h := 0.5 + 0.5*(b-a)/k
	if h < 0 {
		h = 0
	} else if h > 1 {
		h = 1
	}
	return h*a + (1-h)*b - k*h*(1-h)
}

func refNormal(sdf SDF, p Vector) Vector {
	const e = 1e-5
	x, y, z := p.X, p.Y, p.Z
	return Vector{
		sdf.Evaluate(Vector{x - e, y, z}) - sdf.Evaluate(Vector{x + e, y, z}),
		sdf.Evaluate(Vector{x, y - e, z}) - sdf.Evaluate(Vector{x, y + e, z}),
		sdf.Evaluate(Vector{x, y, z - e}) - sdf.Evaluate(Vector{x, y, z + e}),
	}.Normalize()
}

// refIntersect finds the first surface crossing of an SDF along a ray by
// coarse sphere-marching followed by bisection, independent of SDFShape's
// own marching implementation.
func refIntersect(sdf SDF, ray Ray, tMax float64) (Vector, bool) {
	const steps = 4096
	dt := tMax / steps
	t := 0.0
	prev := sdf.Evaluate(ray.Position(t))
	for i := 0; i < steps; i++ {
		t += dt
		d := sdf.Evaluate(ray.Position(t))
		if !math.IsNaN(d) && !math.IsInf(d, 0) && prev >= 0 && d < 0 {
			lo, hi := t-dt, t
			for j := 0; j < 80; j++ {
				mid := (lo + hi) / 2
				if sdf.Evaluate(ray.Position(mid)) > 0 {
					lo = mid
				} else {
					hi = mid
				}
			}
			return ray.Position((lo + hi) / 2), true
		}
		prev = d
	}
	return Vector{}, false
}

func vectorsClose(a, b Vector, tol float64) bool {
	return math.Abs(a.X-b.X) <= tol && math.Abs(a.Y-b.Y) <= tol && math.Abs(a.Z-b.Z) <= tol
}

var (
	redMaterial   = DiffuseMaterial(HexColor(0xff0000))
	blueMaterial  = DiffuseMaterial(HexColor(0x0000ff))
	greenMaterial = DiffuseMaterial(HexColor(0x00ff00))
)

func boxIsFinite(t *testing.T, box Box, label string) {
	comps := []float64{box.Min.X, box.Min.Y, box.Min.Z, box.Max.X, box.Max.Y, box.Max.Z}
	for _, c := range comps {
		if math.IsNaN(c) || math.IsInf(c, 0) {
			t.Fatalf("%s: bounding box has non-finite component: %+v", label, box)
		}
	}
}

// TestSmoothUnionTwoSpheresDistance compares NewSmoothUnionSDF against an
// independent distance reference on a dense point grid.
func TestSmoothUnionTwoSpheresDistance(t *testing.T) {
	const k = 0.5
	c1, c2 := V(-0.6, 0, 0), V(0.6, 0, 0)
	const r = 1.0
	a := NewTransformSDF(NewSphereSDF(r), Translate(c1))
	b := NewTransformSDF(NewSphereSDF(r), Translate(c2))
	smooth := NewSmoothUnionSDF(k, a, b)

	for x := -3.0; x <= 3.0; x += 0.25 {
		for y := -3.0; y <= 3.0; y += 0.25 {
			for z := -3.0; z <= 3.0; z += 0.25 {
				p := V(x, y, z)
				da := refSphereDistance(p, c1, r)
				db := refSphereDistance(p, c2, r)
				want := refSmoothMin(da, db, k)
				got := smooth.Evaluate(p)
				if math.Abs(got-want) > 1e-12 {
					t.Fatalf("Evaluate%v = %v, want %v (da=%v db=%v)", p, got, want, da, db)
				}
			}
		}
	}
}

// TestSmoothUnionKZeroEqualsHardUnion verifies that k=0 and non-finite k
// reproduce the exact hard-union minimum semantics of UnionSDF.
func TestSmoothUnionKZeroEqualsHardUnion(t *testing.T) {
	c1, c2 := V(-0.6, 0, 0), V(0.6, 0, 0)
	a := NewTransformSDF(NewSphereSDF(1), Translate(c1))
	b := NewTransformSDF(NewSphereSDF(1), Translate(c2))
	hard := NewUnionSDF(a, b)

	for _, k := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		smooth := NewSmoothUnionSDF(k, a, b)
		for x := -3.0; x <= 3.0; x += 0.4 {
			for y := -3.0; y <= 3.0; y += 0.4 {
				for z := -3.0; z <= 3.0; z += 0.4 {
					p := V(x, y, z)
					got := smooth.Evaluate(p)
					want := hard.Evaluate(p)
					if got != want {
						t.Fatalf("k=%v Evaluate%v = %v, hard union = %v", k, p, got, want)
					}
				}
			}
		}
	}
}

// TestSmoothUnionExactOutsideBlendRange checks that beyond the blending band
// the smooth minimum is exactly the plain minimum, and that the blend never
// dips more than k/4 below the hard union.
func TestSmoothUnionBlendBounds(t *testing.T) {
	const k = 0.8
	smooth := NewSmoothUnionSDF(k, NewSphereSDF(1), NewTransformSDF(NewSphereSDF(1), Translate(V(3, 0, 0))))
	hard := NewUnionSDF(NewSphereSDF(1), NewTransformSDF(NewSphereSDF(1), Translate(V(3, 0, 0))))
	for x := -4.0; x <= 4.0; x += 0.1 {
		p := V(x, 1.5, 0)
		got := smooth.Evaluate(p)
		want := hard.Evaluate(p)
		if got < want-k/4-1e-12 || got > want+1e-12 {
			t.Fatalf("blend out of bounds at %v: smooth=%v hard=%v", p, got, want)
		}
	}
	// Far from both surfaces where |a-b| >= k: exact minimum.
	p := V(10, 10, 10)
	if smooth.Evaluate(p) != hard.Evaluate(p) {
		t.Fatalf("expected exact minimum outside blend range at %v", p)
	}
}

// TestSmoothUnionBoundingBox verifies the k/4 outward expansion, finiteness
// and containment of every negative-distance point.
func TestSmoothUnionBoundingBox(t *testing.T) {
	const k = 0.6
	c1, c2 := V(-0.6, 0, 0), V(0.6, 0, 0)
	a := NewTransformSDF(NewSphereSDF(1), Translate(c1))
	b := NewTransformSDF(NewSphereSDF(1), Translate(c2))
	smooth := NewSmoothUnionSDF(k, a, b)
	box := smooth.BoundingBox()
	boxIsFinite(t, box, "smooth union")

	hardBox := NewUnionSDF(a, b).BoundingBox()
	e := k / 4
	if !vectorsClose(box.Min, hardBox.Min.SubScalar(e), 1e-12) ||
		!vectorsClose(box.Max, hardBox.Max.AddScalar(e), 1e-12) {
		t.Fatalf("box = %+v, want hard box %+v expanded by %v", box, hardBox, e)
	}

	// Every solid point must lie inside the reported box.
	for i := 0; i < 2000; i++ {
		p := V((float64(i%17)-8)*0.25, (float64((i/17)%17)-8)*0.25, (float64((i/289)%17)-8)*0.25)
		if smooth.Evaluate(p) < 0 && !box.Contains(p) {
			t.Fatalf("solid point %v outside bounding box %+v", p, box)
		}
	}

	// A non-finite / zero k box must equal the hard-union box exactly.
	for _, kk := range []float64{0, math.NaN(), math.Inf(1)} {
		s := NewSmoothUnionSDF(kk, a, b).(*SmoothUnionSDF)
		bk := s.BoundingBox()
		boxIsFinite(t, bk, "degenerate-k smooth union")
		if !vectorsClose(bk.Min, hardBox.Min, 1e-12) || !vectorsClose(bk.Max, hardBox.Max, 1e-12) {
			t.Fatalf("k=%v box = %+v, want %+v", kk, bk, hardBox)
		}
	}
}

// TestEmptySDFNodes checks that item-less combination nodes never produce an
// unbounded box, that their boxes reject every ray, and that Evaluate returns
// +Inf.
func TestEmptySDFNodes(t *testing.T) {
	nodes := map[string]SDF{
		"union":    NewUnionSDF(),
		"diff":     NewDifferenceSDF(),
		"inter":    NewIntersectionSDF(),
		"smooth":   NewSmoothUnionSDF(0.5),
		"smooth-0": NewSmoothUnionSDF(0),
	}
	ray := Ray{V(0, 0, 5), V(0, 0, -1)}
	for name, node := range nodes {
		box := node.BoundingBox()
		boxIsFinite(t, box, name+" empty")
		t1, t2 := box.Intersect(ray)
		if t2 >= t1 {
			t.Fatalf("%s: empty box reports a ray hit: t1=%v t2=%v box=%+v", name, t1, t2, box)
		}
		if d := node.Evaluate(V(0, 0, 0)); !math.IsInf(d, 1) {
			t.Fatalf("%s: empty Evaluate = %v, want +Inf", name, d)
		}
		shape := NewSDFShape(node, redMaterial)
		if hit := shape.Intersect(ray); hit.Ok() {
			t.Fatalf("%s: empty node reports a shape hit", name)
		}
	}

	// Empty nodes embedded inside other nodes and transforms stay well-behaved.
	const ek = 0.4
	combined := NewSmoothUnionSDF(ek, NewUnionSDF(), NewSphereSDF(1))
	wantBox := NewSphereSDF(1).BoundingBox()
	wantBox = Box{wantBox.Min.SubScalar(ek / 4), wantBox.Max.AddScalar(ek / 4)}
	if combined.BoundingBox() != wantBox {
		t.Fatalf("empty child should be ignored in bounding box: got %+v want %+v", combined.BoundingBox(), wantBox)
	}
	wrapped := NewTransformSDF(NewDifferenceSDF(), Rotate(V(1, 0, 0), Radians(37)))
	box := wrapped.BoundingBox()
	boxIsFinite(t, box, "transformed empty")
	if t1, t2 := box.Intersect(ray); t2 >= t1 {
		t.Fatalf("transformed empty box reports a ray hit")
	}
	if !math.IsInf(wrapped.Evaluate(V(1, 2, 3)), 1) {
		t.Fatalf("transformed empty Evaluate should be +Inf")
	}
}

// TestNegativeSDFSizes verifies negative and NaN dimensions never yield an
// unbounded bounding box, contain no solid points, and never panic or report
// phantom hits during evaluation or intersection.
func TestNegativeSDFSizes(t *testing.T) {
	sdfs := map[string]SDF{
		"neg-sphere":   NewSphereSDF(-1),
		"neg-cube":     NewCubeSDF(V(-2, -2, -2)),
		"neg-cyl":      NewCylinderSDF(-0.5, -1),
		"nan-sphere":   NewSphereSDF(math.NaN()),
		"nan-cube":     NewCubeSDF(V(math.NaN(), 1, 1)),
		"neg-smooth-a": NewSmoothUnionSDF(0.3, NewSphereSDF(-1), NewSphereSDF(-0.5)),
	}
	ray := Ray{V(0, 0, 5), V(0, 0, -1)}
	for name, sdf := range sdfs {
		box := sdf.BoundingBox()
		boxIsFinite(t, box, name)
		// A negative/NaN-size solid contains no points: distance must never
		// be negative or NaN (NaN children are treated as infinitely far).
		for i := 0; i < 100; i++ {
			p := V(float64(i%10)-5, float64((i/10)%10)-5, 0)
			d := sdf.Evaluate(p)
			if math.IsNaN(d) || d < 0 {
				t.Fatalf("%s at %v: distance %v should be non-negative and finite", name, p, d)
			}
		}
		if hit := NewSDFShape(sdf, redMaterial).Intersect(ray); hit.Ok() {
			t.Fatalf("%s: negative/NaN-size node reports a hit", name)
		}
	}
}

// TestInfiniteSDFSizes verifies infinite dimensions are clamped into a finite
// bounding box instead of leaking an unbounded box, and remain panic-free.
func TestInfiniteSDFSizes(t *testing.T) {
	sdfs := map[string]SDF{
		"inf-cube":   NewCubeSDF(V(math.Inf(1), 1, 1)),
		"inf-sphere": NewSphereSDF(math.Inf(1)),
		"inf-cyl":    NewCylinderSDF(math.Inf(1), math.Inf(-1)),
		"inf-smooth": NewSmoothUnionSDF(0.3, NewCubeSDF(V(math.Inf(1), 1, 1)), NewSphereSDF(1)),
	}
	ray := Ray{V(0, 0, 5), V(0, 0, -1)}
	for name, sdf := range sdfs {
		box := sdf.BoundingBox()
		boxIsFinite(t, box, name)
		for i := 0; i < 20; i++ {
			p := V(float64(i%5)-2, float64((i/5)%5)-2, 0)
			if d := sdf.Evaluate(p); math.IsNaN(d) {
				t.Fatalf("%s at %v: NaN distance after clamping", name, p)
			}
		}
		// Intersection must stay panic-free regardless of outcome.
		_ = NewSDFShape(sdf, redMaterial).Intersect(ray)
	}
}

// TestSmoothUnionDeepNesting ensures deep nesting does not blow the stack and
// stays correct; same-k nesting is flattened, mixed-k nesting still works.
func TestSmoothUnionDeepNesting(t *testing.T) {
	const depth = 5000
	var sdf SDF = NewSphereSDF(1)
	for i := 0; i < depth; i++ {
		sdf = NewSmoothUnionSDF(0.3, sdf, NewSphereSDF(0.001))
	}
	root := sdf.(*SmoothUnionSDF)
	if len(root.Items) != depth+1 {
		t.Fatalf("same-k nesting not flattened: got %d items, want %d", len(root.Items), depth+1)
	}
	box := root.BoundingBox()
	boxIsFinite(t, box, "flattened deep union")

	var mixed SDF = NewSphereSDF(1)
	for i := 0; i < 2000; i++ {
		mixed = NewSmoothUnionSDF(float64(i%3)+0.1, mixed, NewSphereSDF(0.001))
	}
	if d := mixed.Evaluate(V(0, 0, 0)); d >= 0 || math.IsNaN(d) {
		t.Fatalf("deep mixed-k union interior distance = %v, want negative", d)
	}
	if hit := NewSDFShape(mixed, redMaterial).Intersect(Ray{V(0, 0, 5), V(0, 0, -1)}); !hit.Ok() {
		t.Fatalf("deep mixed-k union should still intersect a ray")
	}
}

// TestSmoothUnionDegenerateKNesting checks that nested smooth unions with
// zero/NaN/Inf k never flatten, keep exact hard-union semantics, produce finite
// boxes, and that all-empty nested trees collapse to the canonical empty box.
func TestSmoothUnionDegenerateKNesting(t *testing.T) {
	a := NewSphereSDF(1)
	b := NewTransformSDF(NewSphereSDF(1), Translate(V(1.2, 0, 0)))
	nested := NewSmoothUnionSDF(0, NewSmoothUnionSDF(0, a, b), a)
	hard := NewUnionSDF(NewUnionSDF(a, b), a)
	for x := -3.0; x <= 4; x += 0.2 {
		for y := -3.0; y <= 3; y += 0.2 {
			p := V(x, y, 0)
			if nested.Evaluate(p) != hard.Evaluate(p) {
				t.Fatalf("nested k=0 at %v: %v != hard %v", p, nested.Evaluate(p), hard.Evaluate(p))
			}
		}
	}
	boxIsFinite(t, nested.BoundingBox(), "nested k=0")

	degenerate := NewSmoothUnionSDF(math.NaN(), nested, NewSmoothUnionSDF(math.Inf(1), a, b))
	boxIsFinite(t, degenerate.BoundingBox(), "NaN/Inf-k nested")

	allEmpty := NewSmoothUnionSDF(0.5, NewSmoothUnionSDF(0.5), NewSmoothUnionSDF(0.5))
	if !sdfIsEmptyBox(allEmpty.BoundingBox()) {
		t.Fatalf("all-empty nested union should yield the empty box, got %+v", allEmpty.BoundingBox())
	}
	if !math.IsInf(allEmpty.Evaluate(V(0, 0, 0)), 1) {
		t.Fatalf("all-empty nested union should evaluate to +Inf")
	}
}

// TestTangentSurfaces checks two spheres that touch at a single point: the
// smooth field stays finite and C1-symmetric there, k=0 keeps the hard seam,
// and normals are well-defined on either side of the contact.
func TestTangentSurfaces(t *testing.T) {
	// Unit spheres centered at +/-1 touch exactly at the origin.
	a := NewTransformSDF(NewSphereSDF(1), Translate(V(-1, 0, 0)))
	b := NewTransformSDF(NewSphereSDF(1), Translate(V(1, 0, 0)))
	hard := NewUnionSDF(a, b)
	smooth := NewSmoothUnionSDF(0.5, a, b)

	if d := smooth.Evaluate(V(0, 0, 0)); math.IsNaN(d) || math.IsInf(d, 0) || d >= 0 {
		t.Fatalf("tangent point smooth distance = %v, want finite negative", d)
	}
	// At a==b the formula reduces to a - k/4.
	if want := hard.Evaluate(V(0, 0, 0)) - 0.5/4; math.Abs(smooth.Evaluate(V(0, 0, 0))-want) > 1e-12 {
		t.Fatalf("tangent bulge = %v, want %v", smooth.Evaluate(V(0, 0, 0)), want)
	}
	// Hard union at the tangent point is exactly 0 with no NaN.
	if d := hard.Evaluate(V(0, 0, 0)); d != 0 {
		t.Fatalf("hard tangent distance = %v, want 0", d)
	}
	for _, p := range []Vector{V(-0.05, 0.05, 0), V(0.05, 0.05, 0), V(0, 0.05, 0)} {
		n1 := refNormal(smooth, p)
		n2 := NewSDFShape(smooth, redMaterial).NormalAt(p)
		if n1.Dot(n2) < 0.9999 {
			t.Fatalf("normal mismatch at %v: ref=%v shape=%v", p, n1, n2)
		}
		if math.IsNaN(n1.X) || math.IsNaN(n2.X) {
			t.Fatalf("NaN normal near tangent at %v", p)
		}
	}
}

// TestSmoothUnionMaterials verifies material selection across differently
// materialed children, including fallback to the shape-level material.
func TestSmoothUnionMaterials(t *testing.T) {
	red := redMaterial
	blue := blueMaterial
	green := greenMaterial
	a := NewMaterialSDF(NewTransformSDF(NewSphereSDF(1), Translate(V(-0.6, 0, 0))), red)
	b := NewMaterialSDF(NewTransformSDF(NewSphereSDF(1), Translate(V(0.6, 0, 0))), blue)
	smooth := NewSmoothUnionSDF(0.4, a, b)
	shape := NewSDFShape(smooth, green)

	cases := []struct {
		p    Vector
		want Color
	}{
		{V(-2, 0, 0), red.Color}, // deep inside left child's region
		{V(2, 0, 0), blue.Color}, // deep inside right child's region
		{V(-0.55, 0, 0), red.Color},
		{V(0.55, 0, 0), blue.Color},
	}
	for _, c := range cases {
		got := shape.MaterialAt(c.p)
		if got.Color != c.want {
			t.Fatalf("MaterialAt%v = %+v, want %+v", c.p, got.Color, c.want)
		}
	}

	// No material attached to any child -> shape-level material wins.
	plain := NewSmoothUnionSDF(0.4, NewSphereSDF(1), NewSphereSDF(1))
	plainShape := NewSDFShape(plain, green)
	if plainShape.MaterialAt(V(0, 0, 0)).Color != green.Color {
		t.Fatalf("expected shape-level fallback material")
	}
	// A materialed child inside a nested smooth union is still reachable.
	nested := NewSmoothUnionSDF(0.2, NewSmoothUnionSDF(0.2, a, NewSphereSDF(0.01)), b)
	nestedShape := NewSDFShape(nested, green)
	if nestedShape.MaterialAt(V(-2, 0, 0)).Color != red.Color {
		t.Fatalf("nested materialed child not reached")
	}

	// Material wrapped inside transforms, scales and hard combiners stays
	// reachable at any nesting depth.
	buried := NewTransformSDF(
		NewScaleSDF(
			NewUnionSDF(
				NewSphereSDF(0.01),
				NewTransformSDF(NewMaterialSDF(NewSphereSDF(0.9), red), Translate(V(-2, 0, 0)))),
			1),
		Identity())
	buriedSmooth := NewSmoothUnionSDF(0.3, buried, NewMaterialSDF(NewSphereSDF(0.5), blue))
	buriedShape := NewSDFShape(buriedSmooth, green)
	if got := buriedShape.MaterialAt(V(-2.5, 0, 0)); got.Color != red.Color {
		t.Fatalf("transform/scale/hard-union nested material not reached: %+v", got.Color)
	}
	if got := buriedShape.MaterialAt(V(0, 0, 0)); got.Color != blue.Color {
		t.Fatalf("sibling material not reached through nested material node: %+v", got.Color)
	}

	// Difference keeps the base child's material.
	diff := NewDifferenceSDF(
		NewMaterialSDF(NewSphereSDF(1.2), red),
		NewMaterialSDF(NewCubeSDF(V(1, 1, 1)), blue))
	if got := NewSDFShape(diff, green).MaterialAt(V(0.9, 0, 0)); got.Color != red.Color {
		t.Fatalf("difference should report the base child's material, got %+v", got.Color)
	}
}

// TestSmoothUnionRayIntersectAndNormal ray-traces the two-sphere blend and
// compares hit points and normals against an independent marcher / gradient.
func TestSmoothUnionRayIntersectAndNormal(t *testing.T) {
	const k = 0.5
	a := NewTransformSDF(NewSphereSDF(1), Translate(V(-0.6, 0, 0)))
	b := NewTransformSDF(NewSphereSDF(1), Translate(V(0.6, 0, 0)))
	smooth := NewSmoothUnionSDF(k, a, b)
	shape := NewSDFShape(smooth, DiffuseMaterial(White))

	rays := []Ray{
		{V(0, 0, 6), V(0, 0, -1)},
		{V(-0.3, 0.2, 5), V(0, 0, -1)},
		{V(0.3, -0.2, 5), V(0, 0, -1)},
		{V(3, 3, 4), V(-0.5, -0.5, -1).Normalize()},
	}
	for _, ray := range rays {
		hit := shape.Intersect(ray)
		if !hit.Ok() {
			t.Fatalf("missed ray %+v that should hit the blended shape", ray)
		}
		hitPoint := ray.Position(hit.T)
		refPoint, ok := refIntersect(smooth, ray, 12)
		if !ok {
			t.Fatalf("reference marcher missed ray %+v", ray)
		}
		if d := hitPoint.Sub(refPoint).Length(); d > 5e-3 {
			t.Fatalf("ray %+v: hit point %v differs from reference %v by %v", ray, hitPoint, refPoint, d)
		}
		// The hit point lies on the zero surface.
		if d := math.Abs(smooth.Evaluate(hitPoint)); d > 5e-2 {
			t.Fatalf("ray %+v: hit point distance %v not on surface", ray, d)
		}
		nShape := shape.NormalAt(hitPoint)
		nRef := refNormal(smooth, hitPoint)
		if nShape.Dot(nRef) < 0.9999 {
			t.Fatalf("ray %+v: normal %v disagrees with reference %v", ray, nShape, nRef)
		}
	}

	// A ray that misses the bounding box never hits.
	miss := Ray{V(0, 10, 0), V(0, 1, 0)}
	if hit := shape.Intersect(miss); hit.Ok() {
		t.Fatalf("ray outside bounding box should miss")
	}
}

// TestDifferenceSphereMinusCube builds the sphere-minus-cube combination and
// checks hit points, normals and distances against the reference marcher.
func TestDifferenceSphereMinusCube(t *testing.T) {
	// Cube half-extent 0.5, sphere radius 1: the cube punches an axis-aligned
	// hole through the sphere.
	base := NewDifferenceSDF(NewSphereSDF(1), NewCubeSDF(V(1, 1, 1)))
	refDist := func(p Vector) float64 {
		return math.Max(refSphereDistance(p, V(0, 0, 0), 1), -NewCubeSDF(V(1, 1, 1)).Evaluate(p))
	}
	shape := NewSDFShape(base, DiffuseMaterial(White))

	for x := -1.5; x <= 1.5; x += 0.25 {
		for y := -1.5; y <= 1.5; y += 0.25 {
			p := V(x, y, 0.3)
			if d := math.Abs(base.Evaluate(p) - refDist(p)); d > 1e-12 {
				t.Fatalf("difference Evaluate%v = %v, want %v", p, base.Evaluate(p), refDist(p))
			}
		}
	}

	// A ray through the subtracted cube column: the outer spherical cap is
	// preserved, so the first hit matches the plain sphere; subtraction only
	// hollows out the middle column (four zero crossings instead of two).
	holeRay := Ray{V(0.1, 0.1, 5), V(0, 0, -1)}
	sphereOnly := NewSDFShape(NewSphereSDF(1), DiffuseMaterial(White))
	sphereHit := sphereOnly.Intersect(holeRay)
	if !sphereHit.Ok() {
		t.Fatalf("plain sphere should be hit")
	}
	holeHit := shape.Intersect(holeRay)
	if !holeHit.Ok() {
		t.Fatalf("ray through the cube column should still hit the difference shape")
	}
	if d := math.Abs(holeHit.T - sphereHit.T); d > 0.02 {
		t.Fatalf("subtraction must preserve the outer spherical cap: t diff=%v", d)
	}
	// The subtracted column itself is empty: field non-negative for |z|<=0.5,
	// while the spherical caps beyond the cube faces stay solid.
	for z := 0.5; z >= -0.5; z -= 0.05 {
		if d := base.Evaluate(V(0.1, 0.1, z)); d < -1e-12 {
			t.Fatalf("subtracted column should be empty, got d=%v at z=%v", d, z)
		}
	}
	for _, z := range []float64{0.8, -0.8} {
		if d := base.Evaluate(V(0.1, 0.1, z)); d >= 0 {
			t.Fatalf("spherical cap at z=%v should remain solid, d=%v", z, d)
		}
	}
	// Counting zero crossings along the ray shows shell-column-shell structure.
	crossings := 0
	prev := base.Evaluate(holeRay.Position(0.0))
	for t := 0.01; t < 7.5; t += 0.01 {
		d := base.Evaluate(holeRay.Position(t))
		if (prev < 0) != (d < 0) {
			crossings++
		}
		prev = d
	}
	if crossings != 4 {
		t.Fatalf("sphere-minus-cube central ray should cross the surface 4 times, got %d", crossings)
	}
	hitRay := Ray{V(0.5, 0, 5), V(0, 0, -1)}
	hit := shape.Intersect(hitRay)
	if !hit.Ok() {
		t.Fatalf("off-axis ray should hit the sphere-minus-cube")
	}
	hitPoint := hitRay.Position(hit.T)
	refPoint, ok := refIntersect(base, hitRay, 10)
	if !ok {
		t.Fatalf("reference marcher missed the difference shape")
	}
	if d := hitPoint.Sub(refPoint).Length(); d > 5e-3 {
		t.Fatalf("difference hit %v vs reference %v differ by %v", hitPoint, refPoint, d)
	}
	nShape := shape.NormalAt(hitPoint)
	nRef := refNormal(base, hitPoint)
	if nShape.Dot(nRef) < 0.9999 {
		t.Fatalf("difference normal %v vs reference %v", nShape, nRef)
	}
	// Where the sphere surface survives subtraction its normal is radial
	// (outward on front caps, inward on back-facing caps).
	if d := math.Abs(nShape.Dot(hitPoint.Normalize())); d < 0.999 {
		t.Fatalf("spherical-cap normal %v not aligned with radial %v (hit %v)", nShape, hitPoint.Normalize(), hitPoint)
	}

	// A smooth-union wrapper around the difference node ray-traces consistently.
	blended := NewSmoothUnionSDF(0.25, base, NewTransformSDF(NewSphereSDF(0.4), Translate(V(0, 0, -1.2))))
	blendedShape := NewSDFShape(blended, DiffuseMaterial(White))
	blendedRay := Ray{V(0, 0, 5), V(0, 0, -1)}
	bh := blendedShape.Intersect(blendedRay)
	if !bh.Ok() {
		t.Fatalf("blended difference should be hit by the central ray")
	}
	bp := blendedRay.Position(bh.T)
	if math.Abs(blended.Evaluate(bp)) > 5e-2 {
		t.Fatalf("blended hit point not on surface: d=%v", blended.Evaluate(bp))
	}
	if n := blendedShape.NormalAt(bp); math.Abs(n.Length()-1) > 1e-9 || math.IsNaN(n.X) {
		t.Fatalf("blended normal degenerate: %v", n)
	}
}

// TestSmoothUnionNestedTransforms verifies Evaluate, bounding box, normals and
// ray intersection stay consistent when the smooth node sits inside nested
// transforms (rotations + translations + non-uniform scale).
func TestSmoothUnionNestedTransforms(t *testing.T) {
	a := NewTransformSDF(NewSphereSDF(0.8), Translate(V(-0.5, 0, 0)))
	b := NewTransformSDF(NewSphereSDF(0.8), Translate(V(0.5, 0, 0)))
	smooth := NewSmoothUnionSDF(0.3, a, b)
	m := Rotate(V(0, 1, 0), Radians(45)).Mul(Translate(V(2, 0, 1))).Mul(Scale(V(1.2, 1, 0.8)))
	world := NewTransformSDF(smooth, m)
	shape := NewSDFShape(world, DiffuseMaterial(White))
	box := shape.BoundingBox()
	boxIsFinite(t, box, "nested-transform smooth union")

	inv := m.Inverse()
	// Evaluate through the transform matches the local field exactly.
	for i := 0; i < 500; i++ {
		p := V(float64(i%23)-11, float64((i/23)%23)-11, float64((i/529)%23)-11).MulScalar(0.25)
		local := inv.MulPosition(p)
		if d := world.Evaluate(p) - smooth.Evaluate(local); math.Abs(d) > 1e-10 {
			t.Fatalf("transformed Evaluate mismatch at %v: %v", p, d)
		}
	}

	ray := Ray{V(1.0, 0, 6), V(0, 0, -1)}
	hit := shape.Intersect(ray)
	if !hit.Ok() {
		t.Fatalf("nested-transform smooth union should be hit")
	}
	p := ray.Position(hit.T)
	if math.Abs(world.Evaluate(p)) > 5e-2 {
		t.Fatalf("nested-transform hit not on surface: d=%v", world.Evaluate(p))
	}
	n := shape.NormalAt(p)
	if math.IsNaN(n.X) || math.Abs(n.Length()-1) > 1e-9 {
		t.Fatalf("nested-transform normal degenerate: %v", n)
	}
}

// TestSDFNormalDegenerateGradient checks that a zero/NaN numerical gradient
// (e.g. at the exact center of a sphere, or the tangent contact seam) yields a
// finite unit normal rather than NaN.
func TestSDFNormalDegenerateGradient(t *testing.T) {
	shape := NewSDFShape(NewSphereSDF(1), DiffuseMaterial(White))
	n := shape.NormalAt(V(0, 0, 0))
	if math.IsNaN(n.X) || math.Abs(n.Length()-1) > 1e-12 {
		t.Fatalf("center normal degenerate: %v", n)
	}
	tangentSDF := NewSmoothUnionSDF(0.5,
		NewTransformSDF(NewSphereSDF(1), Translate(V(-1, 0, 0))),
		NewTransformSDF(NewSphereSDF(1), Translate(V(1, 0, 0))))
	tangent := NewSDFShape(tangentSDF, redMaterial)
	nt := tangent.NormalAt(V(0, 0, 0))
	if math.IsNaN(nt.X) || math.Abs(nt.Length()-1) > 1e-12 {
		t.Fatalf("tangent-point normal degenerate: %v", nt)
	}
}

// TestHardCombinersUnchanged locks in the original hard semantics of union,
// intersection and difference against their pointwise min/max definitions.
func TestHardCombinersUnchanged(t *testing.T) {
	a := NewSphereSDF(1.2)
	b := NewTransformSDF(NewCubeSDF(V(1.4, 1.4, 1.4)), Translate(V(0.4, 0, 0)))
	c := NewTransformSDF(NewSphereSDF(0.6), Translate(V(0.6, 0, 0)))
	union := NewUnionSDF(a, b)
	inter := NewIntersectionSDF(a, b)
	diff := NewDifferenceSDF(a, b, c)
	for x := -2.0; x <= 2.0; x += 0.2 {
		for y := -2.0; y <= 2.0; y += 0.2 {
			for z := -2.0; z <= 2.0; z += 0.2 {
				p := V(x, y, z)
				da, db, dc := a.Evaluate(p), b.Evaluate(p), c.Evaluate(p)
				if got := union.Evaluate(p); got != math.Min(da, db) {
					t.Fatalf("union at %v = %v, want %v", p, got, math.Min(da, db))
				}
				if got := inter.Evaluate(p); got != math.Max(da, db) {
					t.Fatalf("intersection at %v = %v, want %v", p, got, math.Max(da, db))
				}
				wantDiff := math.Max(da, math.Max(-db, -dc))
				if got := diff.Evaluate(p); got != wantDiff {
					t.Fatalf("difference at %v = %v, want %v", p, got, wantDiff)
				}
			}
		}
	}
	// Difference box remains the first child's box.
	if diff.BoundingBox() != a.BoundingBox() {
		t.Fatalf("difference bounding box changed: %+v", diff.BoundingBox())
	}
}

// TestQueriesOutsideBoundingBox checks far-away queries return large positive
// finite distances and rays never register phantom hits.
func TestQueriesOutsideBoundingBox(t *testing.T) {
	smooth := NewSmoothUnionSDF(0.4,
		NewSphereSDF(1),
		NewTransformSDF(NewSphereSDF(1), Translate(V(1.5, 0, 0))))
	for _, p := range []Vector{V(100, 100, 100), V(-1e6, 0, 0), V(0, 0, 50)} {
		d := smooth.Evaluate(p)
		if d <= 0 || math.IsNaN(d) || math.IsInf(d, 0) {
			t.Fatalf("outside-box query %v gave %v, want positive finite", p, d)
		}
	}
	shape := NewSDFShape(smooth, DiffuseMaterial(White))
	for _, ray := range []Ray{
		{V(50, 0, 0), V(1, 0, 0)},
		{V(0, 0, 50), V(0, 0, 1)},
		{V(-50, 20, 30), V(1, -1, -1).Normalize()},
	} {
		hit := shape.Intersect(ray)
		if hit.Ok() {
			t.Fatalf("outward/missing ray %+v should not hit", ray)
		}
	}
}
