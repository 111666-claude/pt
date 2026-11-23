package pt

type Triangle struct {
	Material   *Material
	V1, V2, V3 Vector
	N1, N2, N3 Vector
	T1, T2, T3 Vector
	// UV-driven tangent basis. Tan* are unit tangents orthogonal to the
	// corresponding vertex normals; W* is the per-vertex handedness (+1/-1),
	// with the bitangent recovered as W * (N x T).
	Tan1, Tan2, Tan3 Vector
	W1, W2, W3       float64
	tangentsBuilt    bool
	tangentSmooth    bool
}

func NewTriangle(v1, v2, v3, t1, t2, t3 Vector, material Material) *Triangle {
	t := Triangle{}
	t.V1 = v1
	t.V2 = v2
	t.V3 = v3
	t.T1 = t1
	t.T2 = t2
	t.T3 = t3
	t.Material = &material
	t.FixNormals()
	return &t
}

func (t *Triangle) Vertices() (Vector, Vector, Vector) {
	return t.V1, t.V2, t.V3
}

func (t *Triangle) Compile() {
	t.ensureTangents()
}

func (t *Triangle) BoundingBox() Box {
	min := t.V1.Min(t.V2).Min(t.V3)
	max := t.V1.Max(t.V2).Max(t.V3)
	return Box{min, max}
}

func (t *Triangle) Intersect(r Ray) Hit {
	e1x := t.V2.X - t.V1.X
	e1y := t.V2.Y - t.V1.Y
	e1z := t.V2.Z - t.V1.Z
	e2x := t.V3.X - t.V1.X
	e2y := t.V3.Y - t.V1.Y
	e2z := t.V3.Z - t.V1.Z
	px := r.Direction.Y*e2z - r.Direction.Z*e2y
	py := r.Direction.Z*e2x - r.Direction.X*e2z
	pz := r.Direction.X*e2y - r.Direction.Y*e2x
	det := e1x*px + e1y*py + e1z*pz
	if det > -EPS && det < EPS {
		return NoHit
	}
	inv := 1 / det
	tx := r.Origin.X - t.V1.X
	ty := r.Origin.Y - t.V1.Y
	tz := r.Origin.Z - t.V1.Z
	u := (tx*px + ty*py + tz*pz) * inv
	if u < 0 || u > 1 {
		return NoHit
	}
	qx := ty*e1z - tz*e1y
	qy := tz*e1x - tx*e1z
	qz := tx*e1y - ty*e1x
	v := (r.Direction.X*qx + r.Direction.Y*qy + r.Direction.Z*qz) * inv
	if v < 0 || u+v > 1 {
		return NoHit
	}
	d := (e2x*qx + e2y*qy + e2z*qz) * inv
	if d < EPS {
		return NoHit
	}
	return Hit{t, d, nil}
}

func (t *Triangle) UV(p Vector) Vector {
	u, v, w := t.Barycentric(p)
	n := Vector{}
	n = n.Add(t.T1.MulScalar(u))
	n = n.Add(t.T2.MulScalar(v))
	n = n.Add(t.T3.MulScalar(w))
	return Vector{n.X, n.Y, 0}
}

func (t *Triangle) MaterialAt(p Vector) Material {
	return *t.Material
}

// interpolateUV returns the barycentrically interpolated texture coordinate.
func (t *Triangle) interpolateUV(u, v, w float64) Vector {
	b := Vector{}
	b = b.Add(t.T1.MulScalar(u))
	b = b.Add(t.T2.MulScalar(v))
	b = b.Add(t.T3.MulScalar(w))
	return Vector{b.X, b.Y, 0}
}

// tangentFrame returns the world/local-space tangent basis at barycentric
// coordinates (u, v, w). The tangent is re-orthogonalized against the
// interpolated shading normal so the frame stays orthonormal even after
// per-vertex smoothing or non-uniform transforms.
func (t *Triangle) tangentFrame(u, v, w float64, n Vector) (T, B, N Vector) {
	t.ensureTangents()
	N = n
	T = Vector{}
	T = T.Add(t.Tan1.MulScalar(u))
	T = T.Add(t.Tan2.MulScalar(v))
	T = T.Add(t.Tan3.MulScalar(w))
	handedness := t.W1*u + t.W2*v + t.W3*w
	if handedness < 0 {
		handedness = -1
	} else {
		handedness = 1
	}
	// Gram-Schmidt: remove any component of T along the shading normal.
	T = T.Sub(N.MulScalar(T.Dot(N)))
	if T.Length() < tangentEpsilon || nonFiniteVector(T) {
		// The stored tangent collapsed relative to this normal.
		ft, _ := orthonormalBasis(N)
		T = ft
	}
	T = T.Normalize()
	B = N.Cross(T).MulScalar(handedness).Normalize()
	return T, B, N
}

// tangentToWorld builds the rotation matrix mapping tangent-space vectors into
// local space. Columns are T, B, N, matching the convention of the original
// implementation.
func tangentToWorld(T, B, N Vector) Matrix {
	return Matrix{
		T.X, B.X, N.X, 0,
		T.Y, B.Y, N.Y, 0,
		T.Z, B.Z, N.Z, 0,
		0, 0, 0, 1}
}

func (t *Triangle) NormalAt(p Vector) Vector {
	u, v, w := t.Barycentric(p)
	n := Vector{}
	n = n.Add(t.N1.MulScalar(u))
	n = n.Add(t.N2.MulScalar(v))
	n = n.Add(t.N3.MulScalar(w))
	if n.Length() < tangentEpsilon || nonFiniteVector(n) {
		n = t.geometricNormal()
	}
	n = n.Normalize()
	if t.Material.NormalTexture != nil {
		b := t.interpolateUV(u, v, w)
		ns := t.Material.NormalTexture.NormalSample(b.X, b.Y)
		// A corrupt sample must never poison the shading normal.
		if ns.Length() < tangentEpsilon || nonFiniteVector(ns) {
			ns = Vector{0, 0, 1}
		} else {
			ns = ns.Normalize()
		}
		T, B, N := t.tangentFrame(u, v, w, n)
		n = tangentToWorld(T, B, N).MulDirection(ns)
		if nonFiniteVector(n) {
			n = N
		}
	}
	if t.Material.BumpTexture != nil {
		b := t.interpolateUV(u, v, w)
		bump := t.Material.BumpTexture.BumpSample(b.X, b.Y)
		if !nonFiniteVector(bump) {
			T, B, _ := t.tangentFrame(u, v, w, n)
			n = n.Add(T.MulScalar(bump.X * t.Material.BumpMultiplier))
			n = n.Add(B.MulScalar(bump.Y * t.Material.BumpMultiplier))
		}
	}
	if n.Length() < tangentEpsilon || nonFiniteVector(n) {
		n = t.geometricNormal()
	}
	return n.Normalize()
}

func (t *Triangle) Area() float64 {
	e1 := t.V2.Sub(t.V1)
	e2 := t.V3.Sub(t.V1)
	n := e1.Cross(e2)
	return n.Length() / 2
}

func (t *Triangle) Barycentric(p Vector) (u, v, w float64) {
	v0 := t.V2.Sub(t.V1)
	v1 := t.V3.Sub(t.V1)
	v2 := p.Sub(t.V1)
	d00 := v0.Dot(v0)
	d01 := v0.Dot(v1)
	d11 := v1.Dot(v1)
	d20 := v2.Dot(v0)
	d21 := v2.Dot(v1)
	d := d00*d11 - d01*d01
	v = (d11*d20 - d01*d21) / d
	w = (d00*d21 - d01*d20) / d
	u = 1 - v - w
	return
}

func (t *Triangle) FixNormals() {
	n := t.geometricNormal()
	zero := Vector{}
	if t.N1 == zero {
		t.N1 = n
	}
	if t.N2 == zero {
		t.N2 = n
	}
	if t.N3 == zero {
		t.N3 = n
	}
}
