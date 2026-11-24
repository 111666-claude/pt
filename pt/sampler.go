package pt

import (
	"math"
	"math/rand"
)

type LightMode int

const (
	LightModeRandom = iota
	LightModeAll
)

type SpecularMode int

const (
	SpecularModeNaive = iota
	SpecularModeFirst
	SpecularModeAll
)

type BounceType int

const (
	BounceTypeAny = iota
	BounceTypeDiffuse
	BounceTypeSpecular
)

type Sampler interface {
	Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color
}

func NewSampler(firstHitSamples, maxBounces int) *DefaultSampler {
	return &DefaultSampler{firstHitSamples, maxBounces, true, true, LightModeRandom, SpecularModeNaive}
}

func NewDirectSampler() *DefaultSampler {
	return &DefaultSampler{1, 0, true, false, LightModeAll, SpecularModeAll}
}

type DefaultSampler struct {
	FirstHitSamples int
	MaxBounces      int
	DirectLighting  bool
	SoftShadows     bool
	LightMode       LightMode
	SpecularMode    SpecularMode
}

func (s *DefaultSampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	return s.sample(scene, ray, true, s.FirstHitSamples, 0, rnd)
}

func (s *DefaultSampler) sample(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	if depth > s.MaxBounces {
		return Black
	}
	hit := scene.Intersect(ray)
	if !hit.Ok() {
		return s.sampleEnvironment(scene, ray)
	}
	info := hit.Info(ray)
	material := info.Material
	result := Black
	// negative or NaN emittance is physically invalid: the surface is
	// treated as non-emissive rather than subtracting energy
	if material.Emittance > 0 {
		if s.DirectLighting && !emission {
			return Black
		}
		result = result.Add(material.Color.MulScalar(material.Emittance * float64(samples)))
	}
	n := int(math.Sqrt(float64(samples)))
	var ma, mb BounceType
	if s.SpecularMode == SpecularModeAll || (depth == 0 && s.SpecularMode == SpecularModeFirst) {
		ma = BounceTypeDiffuse
		mb = BounceTypeSpecular
	} else {
		ma = BounceTypeAny
		mb = BounceTypeAny
	}
	for u := 0; u < n; u++ {
		for v := 0; v < n; v++ {
			for mode := ma; mode <= mb; mode++ {
				fu := (float64(u) + rnd.Float64()) / float64(n)
				fv := (float64(v) + rnd.Float64()) / float64(n)
				newRay, reflected, p := ray.Bounce(&info, fu, fv, mode, rnd)
				if mode == BounceTypeAny {
					p = 1
				}
				if p > 0 && reflected {
					// specular
					indirect := s.sample(scene, newRay, reflected, 1, depth+1, rnd)
					tinted := indirect.Mix(material.Color.Mul(indirect), material.Tint)
					result = result.Add(tinted.MulScalar(p))
				}
				if p > 0 && !reflected {
					// diffuse
					indirect := s.sample(scene, newRay, reflected, 1, depth+1, rnd)
					direct := Black
					if s.DirectLighting {
						direct = s.sampleLights(scene, info.Ray, rnd)
					}
					result = result.Add(material.Color.Mul(direct.Add(indirect)).MulScalar(p))
				}
			}
		}
	}
	return result.DivScalar(float64(n * n))
}

func (s *DefaultSampler) sampleEnvironment(scene *Scene, ray Ray) Color {
	if scene.Texture != nil {
		d := ray.Direction
		u := math.Atan2(d.Z, d.X) + scene.TextureAngle
		v := math.Atan2(d.Y, Vector{d.X, 0, d.Z}.Length())
		u = (u + math.Pi) / (2 * math.Pi)
		v = (v + math.Pi/2) / math.Pi
		return scene.Texture.Sample(u, v)
	}
	return scene.Color
}

func (s *DefaultSampler) sampleLights(scene *Scene, n Ray, rnd *rand.Rand) Color {
	nLights := len(scene.Lights)
	if nLights == 0 {
		return Black
	}

	if s.LightMode == LightModeAll {
		var result Color
		for _, light := range scene.Lights {
			result = result.Add(s.sampleLight(scene, n, rnd, light))
		}
		return result
	} else {
		// pick a random light
		light := scene.Lights[rand.Intn(nLights)]
		return s.sampleLight(scene, n, rnd, light).MulScalar(float64(nLights))
	}
}

func (s *DefaultSampler) sampleLight(scene *Scene, n Ray, rnd *rand.Rand, light Shape) Color {
	if area, ok := light.(AreaLight); ok {
		return s.sampleAreaLight(scene, n, rnd, light, area)
	}
	return s.sampleSphericalLight(scene, n, rnd, light)
}

// sampleAreaLight implements direct lighting for triangle, mesh and cube
// lights using the surface-area formulation:
//
//	L_e * cos(theta_o) * |cos(theta_l)| / (r^2 * pdf)
//
// The PDF is a normalized area-measure density, so the estimator is unbiased
// and energy conserving for any tessellation of the emissive surface.
func (s *DefaultSampler) sampleAreaLight(scene *Scene, n Ray, rnd *rand.Rand, light Shape, area AreaLight) Color {
	sample := area.SampleLight(n.Origin, s.SoftShadows, rnd)
	if sample.PDF <= 0 || sample.R2 <= 0 {
		// zero-area / degenerate light, or a sample on top of the shading point
		return Black
	}

	// direction from the shading point toward the sampled light point
	toLight := sample.Point.Sub(n.Origin)
	distance := math.Sqrt(sample.R2)
	ray := Ray{n.Origin, toLight.DivScalar(distance)}

	// cosine on the receiver side: the light must face the shading normal
	diffuse := ray.Direction.Dot(n.Direction)
	if diffuse <= 0 {
		return Black
	}

	// cosine on the light side. Lights in this engine are two-sided, so the
	// absolute value is used: a back-facing emitting triangle still lights
	// points behind it, matching how emissive surfaces are rendered directly.
	lightCosine := math.Abs(ray.Direction.Dot(sample.Normal))
	if lightCosine <= 0 {
		return Black
	}

	// visibility: the first surface hit along the shadow ray must be at or
	// beyond the sampled light point. A small ray-offset keeps the receiver
	// from shadowing itself; a relative tolerance absorbs floating-point
	// rounding between the Moller-Trumbore hit distance and the sampled
	// point distance (they can differ by ~1e-16 on the light surface).
	bias := shadowBias(n.Origin)
	shadowOrigin := n.Origin.Add(ray.Direction.MulScalar(bias))
	shadowRay := Ray{shadowOrigin, ray.Direction}
	hit := scene.Intersect(shadowRay)
	if !hit.Ok() {
		return Black
	}
	tolerance := shadowTolerance * distance
	if hit.T < distance-bias-tolerance {
		return Black
	}

	// negative emittance contributes nothing; MaterialAt honors textures
	material := sample.Material
	emittance := math.Max(0, material.Emittance)
	if emittance == 0 {
		return Black
	}
	weight := emittance * diffuse * lightCosine / (sample.R2 * sample.PDF)
	return material.Color.MulScalar(weight)
}

// shadowTolerance is the relative distance slack used when comparing a
// shadow ray's hit distance with the sampled light distance.
const shadowTolerance = 1e-7

// shadowBias returns a small distance offset proportional to the scene scale
// at the given position, used to avoid shadow-ray self-intersection.
func shadowBias(p Vector) float64 {
	scale := math.Abs(p.X)
	if math.Abs(p.Y) > scale {
		scale = math.Abs(p.Y)
	}
	if math.Abs(p.Z) > scale {
		scale = math.Abs(p.Z)
	}
	return math.Max(EPS, scale*EPS)
}

// sampleSphericalLight keeps the original bounding-disk / solid-angle
// approximation for sphere lights and any other non-area light shapes.
func (s *DefaultSampler) sampleSphericalLight(scene *Scene, n Ray, rnd *rand.Rand, light Shape) Color {
	// get bounding sphere center and radius
	var center Vector
	var radius float64
	switch t := light.(type) {
	case *Sphere:
		radius = t.Radius
		center = t.Center
	default:
		// get bounding sphere from bounding box
		box := t.BoundingBox()
		radius = box.OuterRadius()
		center = box.Center()
	}

	// get random point in disk
	point := center
	if s.SoftShadows {
		for {
			x := rnd.Float64()*2 - 1
			y := rnd.Float64()*2 - 1
			if x*x+y*y <= 1 {
				l := center.Sub(n.Origin).Normalize()
				u := l.Cross(RandomUnitVector(rnd)).Normalize()
				v := l.Cross(u)
				point = Vector{}
				point = point.Add(u.MulScalar(x * radius))
				point = point.Add(v.MulScalar(y * radius))
				point = point.Add(center)
				break
			}
		}
	}

	// construct ray toward light point
	ray := Ray{n.Origin, point.Sub(n.Origin).Normalize()}

	// get cosine term
	diffuse := ray.Direction.Dot(n.Direction)
	if diffuse <= 0 {
		return Black
	}

	// check for light visibility
	hit := scene.Intersect(ray)
	if !hit.Ok() || hit.Shape != light {
		return Black
	}

	// compute solid angle (hemisphere coverage)
	hyp := center.Sub(n.Origin).Length()
	opp := radius
	theta := math.Asin(opp / hyp)
	adj := opp / math.Tan(theta)
	d := math.Cos(theta) * adj
	r := math.Sin(theta) * adj
	coverage := (r * r) / (d * d)

	// TODO: fix issue where hyp < opp (point inside sphere)
	if hyp < opp {
		coverage = 1
	}
	coverage = math.Min(coverage, 1)

	// get material properties from light
	material := MaterialAt(light, point)
	if material.Emittance <= 0 {
		return Black
	}

	// combine factors
	m := material.Emittance * diffuse * coverage
	return material.Color.MulScalar(m)
}
