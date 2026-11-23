package pt

import "math"

// Tangent basis conventions
// -------------------------
// For every triangle the (u, v) texture coordinates and vertex positions define
// a tangent T and bitangent B in local space such that:
//
//	P = P1 + (u-u1)*T + (v-v1)*B
//
// T and B are re-orthogonalized (Gram-Schmidt) against the geometric normal and
// normalized. The handedness sign W is +1 for a right-handed (T, B, N) frame
// and -1 when mirrored UVs flip the frame, so B = W * (N x T).
//
// All routines degrade gracefully: coincident UVs, degenerate triangles,
// non-finite coordinates and missing UVs fall back to a stable orthonormal
// basis built from the geometric normal, and never produce NaN vectors.

const tangentEpsilon = 1e-12

// nonFiniteVector reports whether any component of v is NaN or infinite.
func nonFiniteVector(v Vector) bool {
	return math.IsNaN(v.X) || math.IsNaN(v.Y) || math.IsNaN(v.Z) ||
		math.IsInf(v.X, 0) || math.IsInf(v.Y, 0) || math.IsInf(v.Z, 0)
}

// orthonormalBasis builds a stable orthonormal frame around a unit normal n.
// The returned tangent t satisfies t.Perp(n), and b = n.Cross(t), so that
// (t, b, n) is right handed. It is used whenever UV data is unusable.
func orthonormalBasis(n Vector) (t, b Vector) {
	axis := n.Abs().MinAxis()
	t = axis.Sub(n.MulScalar(axis.Dot(n)))
	if t.Length() < tangentEpsilon {
		// Extremely unlikely, but guard against a pathological axis anyway.
		axis = V(n.Z+0.5, n.X, n.Y+0.25).Normalize()
		t = axis.Sub(n.MulScalar(axis.Dot(n)))
	}
	t = t.Normalize()
	b = n.Cross(t).Normalize()
	return t, b
}

// faceTangent computes an orthogonal tangent/bitangent pair and the handedness
// sign for a single triangle from its positions and UVs. ok is false when the
// triangle has no usable UV parameterization, in which case a geometric-normal
// fallback basis with handedness +1 is returned.
func faceTangent(v1, v2, v3, uv1, uv2, uv3 Vector, n Vector) (t, b Vector, w float64, ok bool) {
	fallback := func() (Vector, Vector, float64, bool) {
		ft, fb := orthonormalBasis(n)
		return ft, fb, 1, false
	}
	// Reject non-finite input before any division can propagate NaN.
	if nonFiniteVector(v1) || nonFiniteVector(v2) || nonFiniteVector(v3) ||
		nonFiniteVector(uv1) || nonFiniteVector(uv2) || nonFiniteVector(uv3) {
		return fallback()
	}
	duv1x := uv2.X - uv1.X
	duv1y := uv2.Y - uv1.Y
	duv2x := uv3.X - uv1.X
	duv2y := uv3.Y - uv1.Y
	det := duv1x*duv2y - duv2x*duv1y
	// Coincident / collinear UVs leave the parameterization undefined.
	if math.Abs(det) < tangentEpsilon {
		return fallback()
	}
	dp1 := v2.Sub(v1)
	dp2 := v3.Sub(v1)
	inv := 1 / det
	// T = (dp1 * duv2y - dp2 * duv1y) / det
	// B = (dp2 * duv1x - dp1 * duv2x) / det
	rawT := dp1.MulScalar(duv2y).Sub(dp2.MulScalar(duv1y)).MulScalar(inv)
	rawB := dp2.MulScalar(duv1x).Sub(dp1.MulScalar(duv2x)).MulScalar(inv)
	if nonFiniteVector(rawT) || nonFiniteVector(rawB) ||
		rawT.Length() < tangentEpsilon || rawB.Length() < tangentEpsilon {
		return fallback()
	}
	// Gram-Schmidt orthogonalization against the geometric normal.
	t = rawT.Sub(n.MulScalar(rawT.Dot(n)))
	if t.Length() < tangentEpsilon {
		return fallback()
	}
	t = t.Normalize()
	// Handedness: positive when (T x B) points along the geometric normal.
	if rawT.Cross(rawB).Dot(n) < 0 {
		w = -1
	} else {
		w = 1
	}
	b = n.Cross(t).MulScalar(w).Normalize()
	return t, b, w, true
}

// BuildTangents computes a per-face tangent basis for every triangle in the
// mesh. This is the hard-edge mode: each triangle keeps its own basis, so no
// smoothing is performed across shared vertices.
func (m *Mesh) BuildTangents() {
	for _, t := range m.Triangles {
		t.buildFaceTangent()
	}
}

// SmoothTangents accumulates per-triangle tangent frames at shared vertices and
// re-orthogonalizes them against the (possibly already smoothed) vertex
// normals. Vertices are keyed by position, exactly like SmoothNormals, so
// triangles sharing a position share a tangent even when their UVs differ.
// Triangles without a usable UV parameterization are skipped during
// accumulation; a vertex that receives no valid tangent falls back to a
// geometric basis derived from its normal.
func (m *Mesh) SmoothTangents() {
	type accum struct {
		t Vector
		w float64
		n int
	}
	lookup := make(map[Vector]*accum)
	get := func(p Vector) *accum {
		a := lookup[p]
		if a == nil {
			a = &accum{}
			lookup[p] = a
		}
		return a
	}
	// First pass: accumulate face tangents and handedness at each vertex.
	for _, t := range m.Triangles {
		ft, _, w, ok := t.faceFrame()
		if !ok {
			continue
		}
		for _, key := range [3]Vector{t.V1, t.V2, t.V3} {
			a := get(key)
			a.t = a.t.Add(ft.MulScalar(w))
			a.w += w
			a.n++
		}
	}
	// Second pass: orthogonalize against the vertex normal and normalize.
	for _, t := range m.Triangles {
		t.tangentsBuilt = true
		t.tangentSmooth = true
		vertices := [3]Vector{t.V1, t.V2, t.V3}
		normals := [3]Vector{t.N1, t.N2, t.N3}
		tangents := [3]*Vector{&t.Tan1, &t.Tan2, &t.Tan3}
		for i := 0; i < 3; i++ {
			a := lookup[vertices[i]]
			n := normals[i]
			if n.Length() < tangentEpsilon {
				n = t.geometricNormal()
			}
			n = n.Normalize()
			if a == nil || a.n == 0 || a.t.Length() < tangentEpsilon {
				ft, _ := orthonormalBasis(n)
				*tangents[i] = ft
				t.setHandedness(i, 1)
				continue
			}
			tt := a.t.Sub(n.MulScalar(a.t.Dot(n)))
			if tt.Length() < tangentEpsilon {
				ft, _ := orthonormalBasis(n)
				*tangents[i] = ft
				t.setHandedness(i, 1)
				continue
			}
			*tangents[i] = tt.Normalize()
			if a.w < 0 {
				t.setHandedness(i, -1)
			} else {
				t.setHandedness(i, 1)
			}
		}
	}
}

// buildFaceTangent fills the per-vertex tangent slots with a single hard-edge
// face frame.
func (t *Triangle) buildFaceTangent() {
	tt, _, w, _ := t.faceFrame()
	t.Tan1, t.Tan2, t.Tan3 = tt, tt, tt
	t.W1, t.W2, t.W3 = w, w, w
	t.tangentsBuilt = true
	t.tangentSmooth = false
}

// faceFrame returns the orthonormal face tangent, bitangent and handedness for
// the triangle, applying the geometric-normal fallback when UVs are unusable.
func (t *Triangle) faceFrame() (tan, bit Vector, w float64, ok bool) {
	n := t.geometricNormal()
	return faceTangent(t.V1, t.V2, t.V3, t.T1, t.T2, t.T3, n)
}

// geometricNormal returns the normalized winding-order face normal. Degenerate
// triangles (collinear vertices) fall back to an arbitrary stable axis instead
// of producing NaN.
func (t *Triangle) geometricNormal() Vector {
	n := t.V2.Sub(t.V1).Cross(t.V3.Sub(t.V1))
	if n.Length() < tangentEpsilon || nonFiniteVector(n) {
		// Pick an axis least aligned with the longest edge direction.
		e := t.V2.Sub(t.V1)
		if e.Length() < tangentEpsilon {
			e = t.V3.Sub(t.V1)
		}
		if e.Length() < tangentEpsilon || nonFiniteVector(e) {
			return Vector{0, 0, 1}
		}
		n = e.Normalize().Abs().MinAxis()
	}
	return n.Normalize()
}

// setHandedness stores the per-vertex handedness sign in W1..W3.
func (t *Triangle) setHandedness(i int, w float64) {
	switch i {
	case 0:
		t.W1 = w
	case 1:
		t.W2 = w
	case 2:
		t.W3 = w
	}
}

// ensureTangents lazily builds a hard-edge basis. It is only a safety net for
// code paths that call NormalAt without compiling the scene; the normal render
// path builds tangents in Compile before any goroutine starts.
func (t *Triangle) ensureTangents() {
	if !t.tangentsBuilt {
		t.buildFaceTangent()
	}
}
