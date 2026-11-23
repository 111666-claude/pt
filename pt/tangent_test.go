package pt

import (
	"image"
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Test texture: a deterministic normal map whose tangent-space normal is a
// fixed function of the UV coordinate, plus a matching bump channel.
// ---------------------------------------------------------------------------

type testNormalTexture struct {
	ns Vector
}

func (tex *testNormalTexture) Sample(u, v float64) Color      { return White }
func (tex *testNormalTexture) Pow(a float64) Texture          { return tex }
func (tex *testNormalTexture) MulScalar(a float64) Texture    { return tex }
func (tex *testNormalTexture) BumpSample(u, v float64) Vector { return Vector{} }
func (tex *testNormalTexture) NormalSample(u, v float64) Vector {
	return tex.ns.Normalize()
}

func newNormalMaterial(ns Vector) Material {
	m := DiffuseMaterial(White)
	m.NormalTexture = &testNormalTexture{ns}
	return m
}

func approx(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func vectorsApprox(a, b Vector, tol float64) bool {
	d := a.Sub(b).Length()
	return d <= tol
}

func isFinite(v Vector) bool {
	return !math.IsNaN(v.X) && !math.IsNaN(v.Y) && !math.IsNaN(v.Z) &&
		!math.IsInf(v.X, 0) && !math.IsInf(v.Y, 0) && !math.IsInf(v.Z, 0)
}

// squareMesh builds a subdivided square in the XY plane (normal +Z) with a
// direct UV mapping (u=x, v=y). It is the workhorse for the basis tests.
func squareMesh(n int, material Material) *Mesh {
	var tris []*Triangle
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			x0 := float64(i) / float64(n)
			y0 := float64(j) / float64(n)
			x1 := float64(i+1) / float64(n)
			y1 := float64(j+1) / float64(n)
			v00, v10, v11 := V(x0, y0, 0), V(x1, y0, 0), V(x1, y1, 0)
			v01 := V(x0, y1, 0)
			u00, u10, u11 := V(x0, y0, 0), V(x1, y0, 0), V(x1, y1, 0)
			u01 := V(x0, y1, 0)
			tris = append(tris,
				NewTriangle(v00, v10, v11, u00, u10, u11, material),
				NewTriangle(v00, v11, v01, u00, u11, u01, material))
		}
	}
	return NewMesh(tris)
}

// triangleAt finds the mesh triangle containing local point p.
func triangleAt(mesh *Mesh, p Vector) *Triangle {
	for _, tri := range mesh.Triangles {
		u, v, w := tri.Barycentric(p)
		if u >= -1e-9 && v >= -1e-9 && w >= -1e-9 && u <= 1+1e-9 && v <= 1+1e-9 && w <= 1+1e-9 {
			return tri
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// 1. Orthogonality, normalization and expected orientation of the basis
// ---------------------------------------------------------------------------

func TestFaceTangentBasisOrthonormal(t *testing.T) {
	mesh := squareMesh(2, DiffuseMaterial(White))
	mesh.BuildTangents()
	for _, tri := range mesh.Triangles {
		for _, tan := range []Vector{tri.Tan1, tri.Tan2, tri.Tan3} {
			if !isFinite(tan) {
				t.Fatalf("non-finite tangent: %v", tan)
			}
			if math.Abs(tan.Length()-1) > 1e-12 {
				t.Fatalf("tangent not unit length: %v len=%v", tan, tan.Length())
			}
		}
		n := tri.geometricNormal()
		T, B, N := tri.tangentFrame(1.0/3, 1.0/3, 1.0/3, n)
		if math.Abs(T.Dot(N)) > 1e-12 || math.Abs(B.Dot(N)) > 1e-12 || math.Abs(T.Dot(B)) > 1e-12 {
			t.Fatalf("basis not orthogonal: T=%v B=%v N=%v", T, B, N)
		}
		if math.Abs(T.Length()-1) > 1e-12 || math.Abs(B.Length()-1) > 1e-12 {
			t.Fatalf("basis vectors not unit length")
		}
		// Direct mapping on the XY plane: T ~ +X, B ~ +Y, handedness +1.
		if !vectorsApprox(T, V(1, 0, 0), 1e-12) {
			t.Fatalf("expected tangent +X, got %v", T)
		}
		if !vectorsApprox(B, V(0, 1, 0), 1e-12) {
			t.Fatalf("expected bitangent +Y, got %v", B)
		}
		if tri.W1 != 1 || tri.W2 != 1 || tri.W3 != 1 {
			t.Fatalf("expected handedness +1, got %v %v %v", tri.W1, tri.W2, tri.W3)
		}
	}
}

// ---------------------------------------------------------------------------
// 2. Normal-map perturbation in tangent space and its rotation covariance
// ---------------------------------------------------------------------------

// referencePerturbedNormal re-derives the expected world normal independently
// of the production code: build the analytic TBN frame from the rotated square,
// transform the fixed tangent-space sample through it.
func referencePerturbedNormal(t *testing.T, tri *Triangle, p, ns Vector) Vector {
	// Analytic tangent for a square whose UVs track local x/y: world +X/+Y
	// rotated by the same transform as the mesh. Here the mesh is untransformed
	// inside TransformedShape, so the local frame is simply X/Y/Z; we then
	// compare against the TransformedShape result via rotation covariance.
	T := V(1, 0, 0)
	B := V(0, 1, 0)
	N := V(0, 0, 1)
	m := tangentToWorld(T, B, N)
	return m.MulDirection(ns).Normalize()
}

func TestNormalTextureTangentSpaceMapping(t *testing.T) {
	ns := V(0.3, -0.4, 1).Normalize()
	mat := newNormalMaterial(ns)
	mesh := squareMesh(1, mat)
	mesh.Compile()
	tri := mesh.Triangles[0]
	p := V(0.25, 0.25, 0)
	got := tri.NormalAt(p)
	want := referencePerturbedNormal(t, tri, p, ns)
	if !vectorsApprox(got, want, 1e-12) {
		t.Fatalf("tangent-space mapping mismatch:\n got %v\nwant %v", got, want)
	}
	// A flat sample must reproduce the geometric normal.
	flat := newNormalMaterial(V(0, 0, 1))
	mesh2 := squareMesh(1, flat)
	mesh2.Compile()
	if n := mesh2.Triangles[0].NormalAt(V(0.5, 0.5, 0)); !vectorsApprox(n, V(0, 0, 1), 1e-12) {
		t.Fatalf("flat normal sample should equal geometric normal, got %v", n)
	}
}

// castTransformed ray-casts a transformed square from the local +Z side at the
// image of local point p, exercising the real TransformedShape hit path
// (inverse ray, inverse-transpose normal, inside flip).
func castTransformed(t *testing.T, shape Shape, m Matrix, p Vector) (HitInfo, Ray) {
	worldP := m.MulPosition(p)
	approach := m.MulDirection(V(0, 0, 1)).MulScalar(5)
	ray := Ray{worldP.Add(approach), worldP.Sub(worldP.Add(approach)).Normalize()}
	hit := shape.Intersect(ray)
	if !hit.Ok() {
		t.Fatalf("expected a hit at %v", worldP)
	}
	return hit.Info(ray), ray
}

func TestRotatedBoxNormalCovariance(t *testing.T) {
	ns := V(0.5, 0.25, 1).Normalize()
	mat := newNormalMaterial(ns)
	angles := []float64{0, math.Pi / 7, math.Pi / 2, math.Pi, 2.3, -1.1}
	axes := []Vector{V(1, 0, 0), V(0, 1, 0), V(0, 0, 1), V(1, 1, 1).Normalize()}
	for _, angle := range angles {
		for _, axis := range axes {
			rot := Rotate(axis, angle)
			mesh := squareMesh(1, mat)
			shape := NewTransformedShape(mesh, rot)
			shape.Compile()
			// Reference: rotate the untransformed triangle result directly.
			refMesh := squareMesh(1, mat)
			refMesh.Compile()
			localP := V(0.3, 0.6, 0)
			refTri := triangleAt(refMesh, localP)
			if refTri == nil {
				t.Fatalf("reference triangle not found")
			}
			want := rot.MulDirection(refTri.NormalAt(localP))
			info, ray := castTransformed(t, shape, rot, localP)
			got := info.Normal
			if !isFinite(got) {
				t.Fatalf("non-finite normal after rotation axis=%v angle=%v", axis, angle)
			}
			// A front-facing hit normal must oppose the ray direction.
			if got.Dot(ray.Direction) >= 0 {
				t.Fatalf("normal not front-facing axis=%v angle=%v: %v", axis, angle, got)
			}
			if !vectorsApprox(got, want, 1e-10) && !vectorsApprox(got, want.Negate(), 1e-10) {
				t.Fatalf("rotation covariance failed axis=%v angle=%v:\n got %v\nwant %v",
					axis, angle, got, want)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// 3. Mirrored UVs flip handedness but keep the frame orthonormal
// ---------------------------------------------------------------------------

func TestMirroredUVHandedness(t *testing.T) {
	mat := newNormalMaterial(V(0.4, 0.4, 1))
	// Mirror the u coordinate: u -> 1-u reverses the tangent orientation.
	v00, v10, v11 := V(0, 0, 0), V(1, 0, 0), V(1, 1, 0)
	v01 := V(0, 1, 0)
	u00, u10, u11 := V(1, 0, 0), V(0, 0, 0), V(0, 1, 0)
	u01 := V(1, 1, 0)
	tri := NewTriangle(v00, v10, v11, u00, u10, u11, mat)
	tri2 := NewTriangle(v00, v11, v01, u00, u11, u01, mat)
	mesh := NewMesh([]*Triangle{tri, tri2})
	mesh.BuildTangents()
	for _, x := range []*Triangle{tri, tri2} {
		if x.W1 != -1 || x.W2 != -1 || x.W3 != -1 {
			t.Fatalf("mirrored UVs should give handedness -1, got %v %v %v", x.W1, x.W2, x.W3)
		}
		n := x.geometricNormal()
		T, B, N := x.tangentFrame(1.0/3, 1.0/3, 1.0/3, n)
		if math.Abs(T.Dot(N)) > 1e-12 || math.Abs(B.Dot(N)) > 1e-12 || math.Abs(T.Dot(B)) > 1e-12 {
			t.Fatalf("mirrored basis not orthogonal: T=%v B=%v N=%v", T, B, N)
		}
		// Mirroring u means the tangent points along -X.
		if !vectorsApprox(T, V(-1, 0, 0), 1e-12) {
			t.Fatalf("mirrored tangent should be -X, got %v", T)
		}
		if !vectorsApprox(B, V(0, 1, 0), 1e-12) {
			t.Fatalf("mirrored bitangent should stay +Y, got %v", B)
		}
	}
	// Pointwise check: the perturbed normal must equal the analytic mirrored
	// frame applied to the same tangent-space sample.
	got := tri.NormalAt(V(0.25, 0.25, 0))
	m := tangentToWorld(V(-1, 0, 0), V(0, 1, 0), V(0, 0, 1))
	want := m.MulDirection(mat.NormalTexture.NormalSample(0, 0)).Normalize()
	if !vectorsApprox(got, want, 1e-12) {
		t.Fatalf("mirrored normal mismatch:\n got %v\nwant %v", got, want)
	}
}

// ---------------------------------------------------------------------------
// 4. Fallbacks: coincident UVs, degenerate triangle, non-finite data, no UVs
// ---------------------------------------------------------------------------

func TestCoincidentUVFallback(t *testing.T) {
	mat := newNormalMaterial(V(0.2, 0.2, 1))
	z := V(0.5, 0.5, 0) // all UVs identical -> zero determinant
	tri := NewTriangle(V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), z, z, z, mat)
	tri.Compile()
	n := tri.geometricNormal()
	T, B, N := tri.tangentFrame(1.0/3, 1.0/3, 1.0/3, n)
	if math.Abs(T.Dot(N)) > 1e-12 || math.Abs(B.Dot(N)) > 1e-12 || math.Abs(T.Dot(B)) > 1e-12 {
		t.Fatalf("fallback basis not orthogonal")
	}
	if !isFinite(T) || !isFinite(B) {
		t.Fatalf("fallback produced non-finite vectors")
	}
	if got := tri.NormalAt(V(0.5, 0.25, 0)); !isFinite(got) {
		t.Fatalf("NormalAt propagated NaN from coincident UVs")
	}
}

func TestDegenerateTriangleFallback(t *testing.T) {
	mat := newNormalMaterial(V(0.1, 0.1, 1))
	tri := NewTriangle(
		V(0, 0, 0), V(1, 0, 0), V(2, 0, 0), // collinear vertices
		V(0, 0, 0), V(1, 0, 0), V(2, 0, 0), mat)
	tri.Compile()
	n := tri.NormalAt(V(1, 0, 0))
	if !isFinite(n) {
		t.Fatalf("degenerate triangle produced non-finite normal: %v", n)
	}
	if math.Abs(n.Length()-1) > 1e-12 {
		t.Fatalf("degenerate triangle normal not unit length: %v", n)
	}
}

func TestNonFiniteUVFallback(t *testing.T) {
	mat := newNormalMaterial(V(0.1, 0.1, 1))
	bad := V(math.NaN(), math.Inf(1), 0)
	tri := NewTriangle(
		V(0, 0, 0), V(1, 0, 0), V(1, 1, 0),
		V(0, 0, 0), bad, V(1, 1, 0), mat)
	tri.Compile()
	if got := tri.NormalAt(V(0.5, 0.25, 0)); !isFinite(got) {
		t.Fatalf("non-finite UVs leaked into normal: %v", got)
	}
}

func TestMissingUVFallback(t *testing.T) {
	// STL-style triangles carry zero UVs and rely on FixNormals.
	mat := newNormalMaterial(V(0.3, 0.3, 1))
	tri := NewTriangle(
		V(0, 0, 0), V(1, 0, 0), V(1, 1, 0),
		Vector{}, Vector{}, Vector{}, mat)
	tri.Compile()
	n := tri.NormalAt(V(0.5, 0.25, 0))
	if !isFinite(n) || math.Abs(n.Length()-1) > 1e-12 {
		t.Fatalf("missing-UV fallback invalid: %v", n)
	}
}

// Corrupt normal texture sample must not propagate NaN.
type nanTexture struct{}

func (nanTexture) Sample(u, v float64) Color      { return Black }
func (nanTexture) Pow(a float64) Texture          { return nanTexture{} }
func (nanTexture) MulScalar(a float64) Texture    { return nanTexture{} }
func (nanTexture) BumpSample(u, v float64) Vector { return Vector{} }
func (nanTexture) NormalSample(u, v float64) Vector {
	return V(math.NaN(), 0, math.Inf(-1))
}

func TestCorruptNormalSampleFallback(t *testing.T) {
	mat := DiffuseMaterial(White)
	mat.NormalTexture = nanTexture{}
	mesh := squareMesh(1, mat)
	mesh.Compile()
	n := mesh.Triangles[0].NormalAt(V(0.5, 0.5, 0))
	if !isFinite(n) {
		t.Fatalf("NaN texture sample propagated: %v", n)
	}
	if !vectorsApprox(n, V(0, 0, 1), 1e-12) {
		t.Fatalf("corrupt sample should fall back to geometric normal, got %v", n)
	}
}

// ---------------------------------------------------------------------------
// 5. Normal flip: reversed winding flips the geometric normal and frame
// ---------------------------------------------------------------------------

func TestNormalFlipWinding(t *testing.T) {
	mat := newNormalMaterial(V(0, 0, 1))
	forward := NewTriangle(V(0, 0, 0), V(1, 0, 0), V(1, 1, 0),
		V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), mat)
	reversed := NewTriangle(V(0, 0, 0), V(1, 1, 0), V(1, 0, 0),
		V(0, 0, 0), V(1, 1, 0), V(1, 0, 0), mat)
	forward.Compile()
	reversed.Compile()
	nf := forward.NormalAt(V(0.5, 0.25, 0))
	nr := reversed.NormalAt(V(0.5, 0.25, 0))
	if !vectorsApprox(nf, V(0, 0, 1), 1e-12) {
		t.Fatalf("forward winding normal wrong: %v", nf)
	}
	if !vectorsApprox(nr, V(0, 0, -1), 1e-12) {
		t.Fatalf("reversed winding normal wrong: %v", nr)
	}
}

// ---------------------------------------------------------------------------
// 6. TransformedShape: inverse-transpose, negative scale, mirroring
// ---------------------------------------------------------------------------

func TestNegativeScaleMirror(t *testing.T) {
	ns := V(0.4, -0.3, 1).Normalize()
	mat := newNormalMaterial(ns)
	cases := []Matrix{
		Scale(V(-1, 1, 1)),
		Scale(V(1, -1, 1)),
		Scale(V(1, 1, -1)),
		Scale(V(-2, 3, 1)),
		Scale(V(0.5, -0.25, 4)),
		Rotate(V(0.3, 0.7, 0.2), 0.9).Scale(V(-1, 1, 1)),
	}
	for i, m := range cases {
		mesh := squareMesh(2, mat)
		shape := NewTransformedShape(mesh, m)
		shape.Compile()
		localP := V(0.3, 0.6, 0)
		info, ray := castTransformed(t, shape, m, localP)
		n := info.Normal
		if !isFinite(n) {
			t.Fatalf("case %d: non-finite normal %v", i, n)
		}
		if math.Abs(n.Length()-1) > 1e-10 {
			t.Fatalf("case %d: normal not unit length %v", i, n)
		}
		if n.Dot(ray.Direction) >= 0 {
			t.Fatalf("case %d: normal not front-facing %v", i, n)
		}
		// Reference: inverse-transpose transform of the local perturbed normal.
		refMesh := squareMesh(2, mat)
		refMesh.Compile()
		refTri := triangleAt(refMesh, localP)
		if refTri == nil {
			t.Fatalf("case %d: reference triangle not found", i)
		}
		localN := refTri.NormalAt(localP)
		want := m.Inverse().Transpose().MulDirection(localN)
		if !vectorsApprox(n, want, 1e-9) && !vectorsApprox(n, want.Negate(), 1e-9) {
			t.Fatalf("case %d inverse-transpose mismatch:\n got %v\nwant %v", i, n, want)
		}
	}
}

// ---------------------------------------------------------------------------
// 7. Smooth tangents: shared vertices get a continuous frame across a seam
// ---------------------------------------------------------------------------

func TestSmoothTangentsSeamContinuity(t *testing.T) {
	mesh := squareMesh(4, DiffuseMaterial(White))
	mesh.SmoothTangents()
	// All vertices on the grid share one normal (+Z); interpolated tangents at
	// any interior point must be continuous and equal to +X.
	for _, tri := range mesh.Triangles {
		if !tri.tangentSmooth {
			t.Fatalf("expected smooth tangent flag")
		}
		n := tri.geometricNormal()
		T, B, N := tri.tangentFrame(0.5, 0.25, 0.25, n)
		if !vectorsApprox(T, V(1, 0, 0), 1e-10) || !vectorsApprox(B, V(0, 1, 0), 1e-10) {
			t.Fatalf("seam discontinuity: T=%v B=%v N=%v", T, B, N)
		}
		if !vectorsApprox(N, V(0, 0, 1), 1e-12) {
			t.Fatalf("smoothed normal drifted: %v", N)
		}
	}
	// A folded mesh (two perpendicular squares sharing an edge) must still
	// produce orthonormal, finite frames at the shared vertices.
	folded := foldedMesh(DiffuseMaterial(White))
	folded.SmoothTangents()
	for _, tri := range folded.Triangles {
		for _, tan := range []Vector{tri.Tan1, tri.Tan2, tri.Tan3} {
			if !isFinite(tan) || math.Abs(tan.Length()-1) > 1e-10 {
				t.Fatalf("folded mesh tangent invalid: %v", tan)
			}
		}
	}
}

// foldedMesh builds two unit squares perpendicular along the x=1 edge, both
// parameterized so the shared edge has matching UVs.
func foldedMesh(material Material) *Mesh {
	a := NewTriangle(V(0, 0, 0), V(1, 0, 0), V(1, 1, 0),
		V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), material)
	b := NewTriangle(V(0, 0, 0), V(1, 1, 0), V(0, 1, 0),
		V(0, 0, 0), V(1, 1, 0), V(0, 1, 0), material)
	c := NewTriangle(V(1, 0, 0), V(1, 0, 1), V(1, 1, 1),
		V(1, 0, 0), V(2, 0, 0), V(2, 1, 0), material)
	d := NewTriangle(V(1, 0, 0), V(1, 1, 1), V(1, 1, 0),
		V(1, 0, 0), V(2, 1, 0), V(1, 1, 0), material)
	return NewMesh([]*Triangle{a, b, c, d})
}

func TestSmoothTangentsWithBadFaces(t *testing.T) {
	mat := DiffuseMaterial(White)
	good := squareMesh(1, mat)
	z := Vector{}
	bad := NewTriangle(V(2, 0, 0), V(3, 0, 0), V(3, 1, 0), z, z, z, mat)
	mesh := NewMesh(append(good.Triangles, bad))
	mesh.SmoothTangents()
	for _, tri := range mesh.Triangles {
		n := tri.NormalAt(tri.V1.Add(tri.V2).Add(tri.V3).MulScalar(1.0 / 3))
		if !isFinite(n) {
			t.Fatalf("smooth build with bad face produced NaN normal")
		}
	}
}

// ---------------------------------------------------------------------------
// 8. Transform updates stored tangents and flips handedness on mirroring
// ---------------------------------------------------------------------------

func TestTransformTangents(t *testing.T) {
	mesh := squareMesh(1, DiffuseMaterial(White))
	mesh.BuildTangents()
	rot := Rotate(V(0, 0, 1), math.Pi/2)
	mesh.Transform(rot)
	tri := mesh.Triangles[0]
	// +X tangent rotated 90deg around Z (right-hand rule) becomes -Y.
	if !vectorsApprox(tri.Tan1, V(0, -1, 0), 1e-10) {
		t.Fatalf("tangent not transformed: %v", tri.Tan1)
	}
	if tri.W1 != 1 {
		t.Fatalf("rotation must not flip handedness")
	}
	// Mirroring flips the handedness sign.
	mesh2 := squareMesh(1, DiffuseMaterial(White))
	mesh2.BuildTangents()
	if w := mesh2.Triangles[0].W1; w != 1 {
		t.Fatalf("setup: expected w=1 got %v", w)
	}
	mesh2.Transform(Scale(V(-1, 1, 1)))
	if w := mesh2.Triangles[0].W1; w != -1 {
		t.Fatalf("mirror should flip handedness, got %v", w)
	}
}

// ---------------------------------------------------------------------------
// 9. ColorTexture still satisfies the Texture interface and image plumbing
// ---------------------------------------------------------------------------

func TestColorTextureNormalSample(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 2, 2))
	// Flat-up normal encoded as RGB (128,128,255) -> roughly (0,0,1) after decode.
	for i := range im.Pix {
		im.Pix[i] = 255
	}
	tex := NewTexture(im)
	ns := tex.NormalSample(0.5, 0.5)
	if !isFinite(ns) {
		t.Fatalf("texture normal sample non-finite: %v", ns)
	}
	if approx(ns.Length(), 1, 1e-12) == false {
		t.Fatalf("texture normal not unit length: %v", ns)
	}
}

// ---------------------------------------------------------------------------
// 10. End-to-end: a rotated textured mesh renders finite colors through a
// TransformedShape hit (covers the Info/inside path).
// ---------------------------------------------------------------------------

func TestTransformedShapeHitFinite(t *testing.T) {
	ns := V(0.2, -0.6, 1).Normalize()
	mat := newNormalMaterial(ns)
	mesh := squareMesh(1, mat)
	rot := Rotate(V(1, 0, 0), math.Pi/4)
	shape := NewTransformedShape(mesh, rot)
	shape.Compile()
	// Ray aimed at the square center from above.
	worldCenter := rot.MulPosition(V(0.5, 0.5, 0))
	ray := Ray{worldCenter.Add(V(0, 0, 2)), V(0, 0, -1)}
	hit := shape.Intersect(ray)
	if !hit.Ok() {
		t.Fatalf("expected a hit")
	}
	info := hit.Info(ray)
	if !isFinite(info.Normal) || math.Abs(info.Normal.Length()-1) > 1e-10 {
		t.Fatalf("hit normal invalid: %v", info.Normal)
	}
	if !isFinite(info.Position) {
		t.Fatalf("hit position non-finite")
	}
}

// ---------------------------------------------------------------------------
// 11. Rotated box: per-face pointwise comparison against an analytic TBN
// reference built independently of the production code.
// ---------------------------------------------------------------------------

// boxMesh builds a unit box centered at the origin with outward winding and a
// planar per-face UV parameterization whose u/v axes form a right-handed frame
// with the face normal.
func boxMesh(material Material) *Mesh {
	h := 0.5
	type face struct {
		corners [4]Vector
		normal  Vector
	}
	faces := []face{
		{[4]Vector{V(-h, -h, h), V(h, -h, h), V(h, h, h), V(-h, h, h)}, V(0, 0, 1)},
		{[4]Vector{V(h, -h, -h), V(-h, -h, -h), V(-h, h, -h), V(h, h, -h)}, V(0, 0, -1)},
		{[4]Vector{V(h, -h, h), V(h, -h, -h), V(h, h, -h), V(h, h, h)}, V(1, 0, 0)},
		{[4]Vector{V(-h, -h, -h), V(-h, -h, h), V(-h, h, h), V(-h, h, -h)}, V(-1, 0, 0)},
		{[4]Vector{V(h, h, -h), V(-h, h, -h), V(-h, h, h), V(h, h, h)}, V(0, 1, 0)},
		{[4]Vector{V(h, -h, h), V(-h, -h, h), V(-h, -h, -h), V(h, -h, -h)}, V(0, -1, 0)},
	}
	var tris []*Triangle
	for _, f := range faces {
		c := f.corners
		// Guarantee outward winding regardless of corner ordering.
		if c[1].Sub(c[0]).Cross(c[2].Sub(c[0])).Dot(f.normal) < 0 {
			c = [4]Vector{c[0], c[3], c[2], c[1]}
		}
		uAxis, vAxis := orthonormalBasis(f.normal)
		uv := func(p Vector) Vector {
			return V(uAxis.Dot(p)+0.5, vAxis.Dot(p)+0.5, 0)
		}
		tris = append(tris,
			NewTriangle(c[0], c[1], c[2], uv(c[0]), uv(c[1]), uv(c[2]), material),
			NewTriangle(c[0], c[2], c[3], uv(c[0]), uv(c[2]), uv(c[3]), material))
	}
	return NewMesh(tris)
}

func TestRotatedBoxPerFaceReference(t *testing.T) {
	ns := V(0.35, -0.5, 1).Normalize()
	mat := newNormalMaterial(ns)
	rot := Rotate(V(0.4, 1, 0.3), 1.23)
	mesh := boxMesh(mat)
	shape := NewTransformedShape(mesh, rot)
	shape.Compile()
	faceNormals := []Vector{
		V(0, 0, 1), V(0, 0, -1), V(1, 0, 0), V(-1, 0, 0), V(0, 1, 0), V(0, -1, 0),
	}
	for _, n := range faceNormals {
		// Analytic local frame: the box UVs are built from this same TBN, so the
		// perturbed local normal is TBN * ns and the world normal is R * that.
		tAxis, bAxis := orthonormalBasis(n)
		want := rot.MulDirection(tangentToWorld(tAxis, bAxis, n).MulDirection(ns))
		// Cast a ray through the rotated box along the rotated face normal.
		center := rot.MulPosition(Vector{})
		approach := rot.MulDirection(n).MulScalar(3)
		ray := Ray{center.Add(approach), approach.Negate().Normalize()}
		hit := shape.Intersect(ray)
		if !hit.Ok() {
			t.Fatalf("face %v: no hit", n)
		}
		got := hit.Info(ray).Normal
		if !isFinite(got) || math.Abs(got.Length()-1) > 1e-10 {
			t.Fatalf("face %v: invalid normal %v", n, got)
		}
		if !vectorsApprox(got, want, 1e-9) && !vectorsApprox(got, want.Negate(), 1e-9) {
			t.Fatalf("face %v: perturbed normal mismatch:\n got %v\nwant %v", n, got, want)
		}
	}
	// With a flat normal map every face must return exactly its rotated
	// geometric normal — the pure geometric reference.
	flat := newNormalMaterial(V(0, 0, 1))
	flatMesh := boxMesh(flat)
	flatShape := NewTransformedShape(flatMesh, rot)
	flatShape.Compile()
	for _, n := range faceNormals {
		center := rot.MulPosition(Vector{})
		approach := rot.MulDirection(n).MulScalar(3)
		ray := Ray{center.Add(approach), approach.Negate().Normalize()}
		hit := flatShape.Intersect(ray)
		if !hit.Ok() {
			t.Fatalf("flat face %v: no hit", n)
		}
		want := rot.MulDirection(n)
		got := hit.Info(ray).Normal
		if !vectorsApprox(got, want, 1e-9) && !vectorsApprox(got, want.Negate(), 1e-9) {
			t.Fatalf("flat face %v: expected geometric normal, got %v want %v", n, got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// 12. Seam continuity: smooth tangents make perturbed normals continuous
// across a shared edge whose two faces carry different UV parameterizations.
// ---------------------------------------------------------------------------

func TestSeamEdgeContinuity(t *testing.T) {
	mat := newNormalMaterial(V(0.4, 0.25, 1).Normalize())
	v00, v10, v11, v01 := V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), V(0, 1, 0)
	// Face A: standard UVs (u along +X, v along +Y).
	a := NewTriangle(v00, v10, v11, V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), mat)
	// Face B: rotated UVs (u along +Y), giving a different face tangent and
	// negative handedness — a real UV seam along the shared diagonal.
	b := NewTriangle(v00, v11, v01, V(0, 0, 0), V(1, 1, 0), V(1, 0, 0), mat)
	edgePoint := V(0.5, 0.5, 0)

	hard := NewMesh([]*Triangle{a, b})
	hard.BuildTangents()
	hardA := a.NormalAt(edgePoint)
	hardB := b.NormalAt(edgePoint)
	if !isFinite(hardA) || !isFinite(hardB) {
		t.Fatalf("hard-edge normals non-finite")
	}
	if vectorsApprox(hardA, hardB, 1e-10) {
		t.Fatalf("hard-edge seam is unexpectedly continuous: %v", hardA)
	}

	// Fresh copies for the smooth build (triangles hold frame state).
	a2 := NewTriangle(v00, v10, v11, V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), mat)
	b2 := NewTriangle(v00, v11, v01, V(0, 0, 0), V(1, 1, 0), V(1, 0, 0), mat)
	smooth := NewMesh([]*Triangle{a2, b2})
	smooth.SmoothTangents()
	smoothA := a2.NormalAt(edgePoint)
	smoothB := b2.NormalAt(edgePoint)
	if !isFinite(smoothA) || !isFinite(smoothB) {
		t.Fatalf("smooth seam normals non-finite")
	}
	if !vectorsApprox(smoothA, smoothB, 1e-9) {
		t.Fatalf("seam discontinuity after SmoothTangents:\n a %v\n b %v", smoothA, smoothB)
	}
	// The shared vertices themselves must carry identical tangent vectors.
	if !vectorsApprox(a2.Tan1, b2.Tan1, 1e-10) || !vectorsApprox(a2.Tan3, b2.Tan2, 1e-10) {
		t.Fatalf("shared-vertex tangents not unified by smoothing")
	}
}

// Compile must not clobber a frame that was explicitly built with SmoothTangents.
func TestCompilePreservesSmoothTangents(t *testing.T) {
	a := NewTriangle(V(0, 0, 0), V(1, 0, 0), V(1, 1, 0),
		V(0, 0, 0), V(1, 0, 0), V(1, 1, 0), DiffuseMaterial(White))
	b := NewTriangle(V(0, 0, 0), V(1, 1, 0), V(0, 1, 0),
		V(0, 0, 0), V(1, 1, 0), V(0, 1, 0), DiffuseMaterial(White))
	mesh := NewMesh([]*Triangle{a, b})
	mesh.SmoothTangents()
	want := a.Tan1
	mesh.Compile()
	if !a.tangentSmooth {
		t.Fatalf("Compile replaced smooth tangents with hard-edge frames")
	}
	if !vectorsApprox(a.Tan1, want, 1e-12) {
		t.Fatalf("Compile altered a pre-built smooth tangent: got %v want %v", a.Tan1, want)
	}
}

// Bump mapping uses the same stable frame: a bump along u tilts the normal
// toward the tangent, along v toward the bitangent, with no NaN.
type bumpTexture struct{ bu, bv float64 }

func (tex bumpTexture) Sample(u, v float64) Color   { return White }
func (tex bumpTexture) Pow(a float64) Texture       { return tex }
func (tex bumpTexture) MulScalar(a float64) Texture { return tex }
func (tex bumpTexture) NormalSample(u, v float64) Vector {
	return V(0, 0, 1)
}
func (tex bumpTexture) BumpSample(u, v float64) Vector { return V(tex.bu, tex.bv, 0) }

func TestBumpUsesTangentFrame(t *testing.T) {
	mat := DiffuseMaterial(White)
	mat.BumpTexture = bumpTexture{0.3, 0.2}
	mat.BumpMultiplier = 1
	mesh := squareMesh(1, mat)
	mesh.Compile()
	tri := mesh.Triangles[0]
	n := tri.NormalAt(V(0.25, 0.25, 0))
	want := V(0, 0, 1).Add(V(1, 0, 0).MulScalar(0.3)).Add(V(0, 1, 0).MulScalar(0.2)).Normalize()
	if !isFinite(n) {
		t.Fatalf("bump normal non-finite")
	}
	if !vectorsApprox(n, want, 1e-12) {
		t.Fatalf("bump mismatch:\n got %v\nwant %v", n, want)
	}
	// Mirrored UVs must flip the bump direction along with the tangent.
	mirrored := NewTriangle(
		V(0, 0, 0), V(1, 0, 0), V(1, 1, 0),
		V(1, 0, 0), V(0, 0, 0), V(0, 1, 0), mat)
	mirrored.Compile()
	nm := mirrored.NormalAt(V(0.25, 0.25, 0))
	wantM := V(0, 0, 1).Add(V(-1, 0, 0).MulScalar(0.3)).Add(V(0, 1, 0).MulScalar(0.2)).Normalize()
	if !vectorsApprox(nm, wantM, 1e-12) {
		t.Fatalf("mirrored bump mismatch:\n got %v\nwant %v", nm, wantM)
	}
}
