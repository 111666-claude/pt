package pt

import (
	"math"
	"math/rand"
)

// LightSample is a point sampled on the surface of an emissive shape.
//
// PDF is the probability density with respect to surface area. When it is
// non-zero it is normalized over the whole surface of the light shape.
// R2 is the squared distance from the shading point (passed to SampleLight)
// to the sampled point.
type LightSample struct {
	Point    Vector
	Normal   Vector
	R2       float64
	PDF      float64
	Material Material
}

// AreaLight is a Shape that can be sampled by surface area. Triangle, Mesh
// and Cube lights implement this interface and use the unbiased area
// estimator instead of the bounding-disk approximation used for other
// shapes.
type AreaLight interface {
	Shape
	// LightArea returns the total emissive surface area. It is zero for
	// degenerate lights that carry no sampleable surface.
	LightArea() float64
	// SampleLight picks a point on the emissive surface. When softShadows
	// is false the light's centroid is used deterministically, producing
	// hard shadows without changing the expected energy. A zero PDF means
	// the light cannot be sampled (zero emissive area); otherwise the PDF
	// is a normalized area-measure density.
	SampleLight(origin Vector, softShadows bool, rnd *rand.Rand) LightSample
}

// shapeEmittance reports the (non-negative) emittance of a shape. Negative
// emittance values are treated as "not a light" everywhere in the sampler.
func shapeEmittance(shape Shape) float64 {
	return math.Max(0, shape.MaterialAt(Vector{}).Emittance)
}

// sampleTriangleUniform returns a uniformly distributed point on triangle t
// together with its geometric normal and its area. The area measure density
// of the returned point is 1/area.
func sampleTriangleUniform(t *Triangle, rnd *rand.Rand) (point, normal Vector, area float64) {
	u1 := rnd.Float64()
	u2 := rnd.Float64()
	su1 := math.Sqrt(u1)
	a := 1 - su1
	b := su1 * (1 - u2)
	c := su1 * u2
	point = t.V1.MulScalar(a).Add(t.V2.MulScalar(b)).Add(t.V3.MulScalar(c))
	e1 := t.V2.Sub(t.V1)
	e2 := t.V3.Sub(t.V1)
	cross := e1.Cross(e2)
	normal = cross.Normalize()
	area = cross.Length() / 2
	return
}

// LightArea makes Triangle an AreaLight. A triangle with non-positive
// emittance is not a light and therefore reports zero sampleable area.
func (t *Triangle) LightArea() float64 {
	if shapeEmittance(t) <= 0 {
		return 0
	}
	return t.Area()
}

// SampleLight samples the triangle uniformly by area. With softShadows
// disabled the centroid is used deterministically (hard shadow) while the
// PDF still reports the normalized area density, so energy is unchanged.
func (t *Triangle) SampleLight(origin Vector, softShadows bool, rnd *rand.Rand) LightSample {
	area := t.Area()
	if !(area > 0) {
		return LightSample{PDF: 0}
	}
	var point, normal Vector
	if softShadows {
		point, normal, _ = sampleTriangleUniform(t, rnd)
	} else {
		e1 := t.V2.Sub(t.V1)
		e2 := t.V3.Sub(t.V1)
		normal = e1.Cross(e2).Normalize()
		point = t.V1.Add(t.V2).Add(t.V3).MulScalar(1.0 / 3.0)
	}
	return LightSample{
		Point:    point,
		Normal:   normal,
		R2:       point.Sub(origin).LengthSquared(),
		PDF:      1 / area,
		Material: t.MaterialAt(point),
	}
}

// lightTriangle records one emissive triangle of a Mesh and its share of the
// area-weighted sampling distribution.
type lightTriangle struct {
	triangle *Triangle
	area     float64
	cdf      float64 // cumulative probability through this entry, in (0, 1]
}

// buildLightDistribution collects the emissive, non-degenerate triangles of a
// Mesh into an area-weighted cumulative distribution. Zero-area and
// degenerate triangles are skipped: they can never be hit by a ray and must
// not carry any probability mass.
func (m *Mesh) buildLightDistribution() {
	m.lightDirty = false
	m.lightTriangles = nil
	m.lightArea = 0
	var total float64
	for _, t := range m.Triangles {
		if shapeEmittance(t) <= 0 {
			continue
		}
		area := t.Area()
		if !(area > 0) || math.IsInf(area, 0) {
			continue
		}
		total += area
		m.lightTriangles = append(m.lightTriangles, lightTriangle{triangle: t, area: area})
	}
	if total <= 0 {
		m.lightTriangles = nil
		m.lightArea = 0
		return
	}
	cum := 0.0
	centroid := Vector{}
	for i := range m.lightTriangles {
		entry := &m.lightTriangles[i]
		cum += entry.area
		entry.cdf = cum / total
		t := entry.triangle
		centroid = centroid.Add(t.V1.Add(t.V2).Add(t.V3).MulScalar(entry.area / 3.0))
	}
	// guard against accumulated floating point error: the last entry is 1
	m.lightTriangles[len(m.lightTriangles)-1].cdf = 1
	m.lightArea = total
	m.lightCentroid = centroid.DivScalar(total)
}

func (m *Mesh) ensureLightDistribution() {
	if m.lightDirty || m.lightTriangles == nil {
		m.buildLightDistribution()
	}
}

// LightArea makes Mesh an AreaLight. It is zero when the mesh contains no
// emissive triangle with positive area.
func (m *Mesh) LightArea() float64 {
	m.ensureLightDistribution()
	return m.lightArea
}

// SampleLight picks an emissive triangle with probability proportional to its
// area, then a uniform point on that triangle. The area measure PDF of the
// returned point is 1 over the total emissive area and therefore normalized.
func (m *Mesh) SampleLight(origin Vector, softShadows bool, rnd *rand.Rand) LightSample {
	m.ensureLightDistribution()
	if m.lightArea <= 0 || len(m.lightTriangles) == 0 {
		return LightSample{PDF: 0}
	}
	if !softShadows {
		// hard shadows: aim at the area-weighted centroid of the emissive mesh.
		// MaterialAt/NormalAt are unimplemented on Mesh, so the material and
		// geometric normal are taken from an emissive triangle associated with
		// the centroid. Barycentric normal interpolation is avoided because
		// the centroid may sit outside any single triangle.
		entry := m.centroidTriangle()
		tri := entry.triangle
		e1 := tri.V2.Sub(tri.V1)
		e2 := tri.V3.Sub(tri.V1)
		normal := e1.Cross(e2).Normalize()
		point := m.lightCentroid
		return LightSample{
			Point:    point,
			Normal:   normal,
			R2:       point.Sub(origin).LengthSquared(),
			PDF:      1 / m.lightArea,
			Material: tri.MaterialAt(point),
		}
	}
	r := rnd.Float64()
	i := searchCDF(m.lightTriangles, r)
	entry := &m.lightTriangles[i]
	point, normal, _ := sampleTriangleUniform(entry.triangle, rnd)
	return LightSample{
		Point:    point,
		Normal:   normal,
		R2:       point.Sub(origin).LengthSquared(),
		PDF:      1 / m.lightArea,
		Material: entry.triangle.MaterialAt(point),
	}
}

// centroidTriangle returns the distribution entry used for the hard-shadow
// centroid sample: preferably an emissive triangle whose surface contains
// the centroid, otherwise the largest emissive triangle.
func (m *Mesh) centroidTriangle() *lightTriangle {
	best := &m.lightTriangles[0]
	for i := range m.lightTriangles {
		entry := &m.lightTriangles[i]
		t := entry.triangle
		u, v, w := t.Barycentric(m.lightCentroid)
		if u >= 0 && v >= 0 && w >= 0 && u <= 1 && v <= 1 && w <= 1 {
			return entry
		}
		if entry.area > best.area {
			best = entry
		}
	}
	return best
}

// searchCDF returns the first distribution entry whose cumulative probability
// is greater than r (r in [0, 1)).
func searchCDF(entries []lightTriangle, r float64) int {
	lo, hi := 0, len(entries)
	for lo < hi {
		mid := (lo + hi) / 2
		if entries[mid].cdf <= r {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo >= len(entries) {
		lo = len(entries) - 1
	}
	return lo
}

// lightMesh lazily builds the triangle mesh representation used to sample a
// Cube light by area. sync.Once makes the lazy build safe under the
// renderer's concurrent goroutines.
func (c *Cube) lightMesh() *Mesh {
	c.lightOnce.Do(func() {
		c.lightMeshCache = c.Mesh()
	})
	return c.lightMeshCache
}

// LightArea makes Cube an AreaLight.
func (c *Cube) LightArea() float64 {
	return c.lightMesh().LightArea()
}

// SampleLight samples the surface of the cube light uniformly by area.
func (c *Cube) SampleLight(origin Vector, softShadows bool, rnd *rand.Rand) LightSample {
	return c.lightMesh().SampleLight(origin, softShadows, rnd)
}
