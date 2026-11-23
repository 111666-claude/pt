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
	return s.sample(scene, ray, true, s.FirstHitSamples, 0, false, 0, rnd)
}

// specular indicates whether the incoming ray was produced by a
// perfectly specular (reflection/refraction) event, in which case the BSDF
// sampling density is a delta and environment MIS weights must collapse to
// one. bsdfPDF is the solid-angle density the current path assigned to the
// incoming ray; it is zero for primary rays and delta specular rays.
func (s *DefaultSampler) sample(scene *Scene, ray Ray, emission bool, samples, depth int, specular bool, bsdfPDF float64, rnd *rand.Rand) Color {
	if depth > s.MaxBounces {
		return Black
	}
	hit := scene.Intersect(ray)
	if !hit.Ok() {
		return s.sampleEnvironmentPath(scene, ray, specular, bsdfPDF)
	}
	info := hit.Info(ray)
	material := info.Material
	result := Black
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
					// specular: perfect reflection/refraction is a delta
					// event, so the BSDF pdf is not a finite density and
					// environment MIS on the escape branch weights to one.
					indirect := s.sample(scene, newRay, reflected, 1, depth+1, true, 0, rnd)
					tinted := indirect.Mix(material.Color.Mul(indirect), material.Tint)
					result = result.Add(tinted.MulScalar(p))
				}
				if p > 0 && !reflected {
					// diffuse: WeightedBounce is cosine-weighted, so the
					// solid-angle BSDF density is cos(theta)/pi.
					cosTheta := info.Ray.Direction.Dot(newRay.Direction)
					pdf := math.Max(0, cosTheta) / math.Pi
					indirect := s.sample(scene, newRay, reflected, 1, depth+1, false, pdf, rnd)
					direct := Black
					if s.DirectLighting {
						direct = s.sampleLights(scene, info.Ray, rnd)
					}
					// next-event estimation of the environment map; this
					// only exists when the scene enabled EnvSampling and a
					// valid distribution could be built.
					envDirect := s.sampleEnvironmentDirect(scene, info.Ray, rnd)
					result = result.Add(material.Color.Mul(direct.Add(envDirect).Add(indirect)).MulScalar(p))
				}
			}
		}
	}
	return result.DivScalar(float64(n * n))
}

// envEnabled reports whether environment importance sampling can be used.
func (s *DefaultSampler) envEnabled(scene *Scene) bool {
	if !scene.EnvSampling || scene.Texture == nil {
		return false
	}
	return scene.EnvDistribution().Valid()
}

// powerHeuristic is the canonical balance/weight heuristic used to combine
// two non-delta sampling techniques (here BSDF sampling and environment
// map sampling).
func powerHeuristic(pdfA, pdfB float64) float64 {
	if pdfA <= 0 {
		return 0
	}
	if pdfB <= 0 {
		return 1
	}
	a2 := pdfA * pdfA
	return a2 / (a2 + pdfB*pdfB)
}

// sampleEnvironmentPath evaluates an escaped ray against the environment.
// With environment importance sampling enabled, diffuse escape paths are
// combined with the environment next-event estimator via the power
// heuristic; delta specular escapes keep full weight because their BSDF
// density is not representable as a finite pdf.
func (s *DefaultSampler) sampleEnvironmentPath(scene *Scene, ray Ray, specular bool, bsdfPDF float64) Color {
	color := s.sampleEnvironment(scene, ray)
	if !s.envEnabled(scene) || specular {
		return color
	}
	envPDF := scene.EnvDistribution().PDF(ray.Direction)
	if envPDF <= 0 {
		// direction outside the supported distribution (e.g. black texel):
		// only the BSDF technique can have generated it, keep full weight.
		return color
	}
	if bsdfPDF <= 0 {
		return color
	}
	return color.MulScalar(powerHeuristic(bsdfPDF, envPDF))
}

// sampleEnvironmentDirect performs next-event estimation toward the
// environment: it draws an importance-sampled direction, checks that no
// geometry occludes it, and returns the weighted radiance including the
// cosine term, BSDF albedo being applied by the caller like any other
// direct light.
func (s *DefaultSampler) sampleEnvironmentDirect(scene *Scene, n Ray, rnd *rand.Rand) Color {
	if !s.envEnabled(scene) {
		return Black
	}
	dist := scene.EnvDistribution()
	dir, envPDF, _, _ := dist.Sample(rnd)
	if envPDF <= 0 || !isFinite(envPDF) {
		return Black
	}
	cosTheta := n.Direction.Dot(dir)
	if cosTheta <= 0 {
		return Black
	}
	// offset along the normal like the refraction path does, to avoid
	// re-intersecting the emitting surface at t ~ 0
	origin := n.Origin.Add(n.Direction.MulScalar(1e-4))
	shadowRay := Ray{origin, dir}
	if hit := scene.Intersect(shadowRay); hit.Ok() {
		return Black
	}
	u, v := envDirectionToUV(dir, dist.textureAngle)
	le := scene.Texture.Sample(u, v)
	if !isFiniteColor(le) {
		return Black
	}
	// NEE for the irradiance integral L(w) cos(theta) dw, estimated as
	// L cos / pdf_env. (The 1/pi Lambertian factor lives in the surface
	// albedo/BSDF applied by the caller and in bsdfPDF used for the MIS
	// weight; it must not be folded into this light estimator.)
	bsdfPDF := cosTheta / math.Pi
	weight := powerHeuristic(envPDF, bsdfPDF)
	return le.MulScalar(cosTheta * weight / envPDF)
}

func isFiniteColor(c Color) bool {
	return isFinite(c.R) && isFinite(c.G) && isFinite(c.B)
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

	// combine factors
	m := material.Emittance * diffuse * coverage
	return material.Color.MulScalar(m)
}
