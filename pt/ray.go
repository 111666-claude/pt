package pt

import (
	"math"
	"math/rand"
)

type Ray struct {
	Origin, Direction Vector
}

func (r Ray) Position(t float64) Vector {
	return r.Origin.Add(r.Direction.MulScalar(t))
}

func (n Ray) Reflect(i Ray) Ray {
	return Ray{n.Origin, n.Direction.Reflect(i.Direction)}
}

func (n Ray) Refract(i Ray, n1, n2 float64) Ray {
	return Ray{n.Origin, n.Direction.Refract(i.Direction, n1, n2)}
}

func (n Ray) Reflectance(i Ray, n1, n2 float64) float64 {
	return n.Direction.Reflectance(i.Direction, n1, n2)
}

func (r Ray) WeightedBounce(u, v float64, rnd *rand.Rand) Ray {
	radius := math.Sqrt(u)
	theta := 2 * math.Pi * v
	s := r.Direction.Cross(RandomUnitVector(rnd)).Normalize()
	t := r.Direction.Cross(s)
	d := Vector{}
	d = d.Add(s.MulScalar(radius * math.Cos(theta)))
	d = d.Add(t.MulScalar(radius * math.Sin(theta)))
	d = d.Add(r.Direction.MulScalar(math.Sqrt(1 - u)))
	return Ray{r.Origin, d}
}

func (r Ray) ConeBounce(theta, u, v float64, rnd *rand.Rand) Ray {
	return Ray{r.Origin, Cone(r.Direction, theta, u, v, rnd)}
}

// ggxSpecularWeight computes the reflection weight for a microfacet bounce.
// Rays are produced by importance-sampling the GGX NDF (pdf_l = D/4), so the
// Monte Carlo weight f*cosL/pdf for the Cook-Torrance BRDF
// f = D*G2*F/(4*cosV*cosL) simplifies to G2*F/cosV: the NDF cancels out and
// no 1/alpha^2 singularity remains. In the smooth mirror limit G2 -> 1 and
// cosV -> 1, so the weight collapses to the plain Fresnel factor. An
// explicit Reflectivity overrides the Fresnel term. The result is always
// finite and non-negative (clamped to [0, 1]; the clamp only trims rare
// grazing-angle fireflies and costs negligible energy).
func ggxSpecularWeight(material Material, n, i, l Vector) float64 {
	cosThetaV := safeCos(n.Dot(i.Negate()))
	cosThetaL := safeCos(n.Dot(l))
	var f float64
	if material.Reflectivity >= 0 {
		f = material.Reflectivity
	} else {
		f = fresnelSchlick(material.f0(), cosThetaV).MaxComponent()
	}
	f = Clamp(f, 0, 1)
	if material.Roughness <= 0 {
		// Stable specular limit: exact mirror, no geometric attenuation.
		return f
	}
	if cosThetaV < 1e-6 {
		// Grazing degeneracy: keep the Fresnel response rather than
		// dividing by a vanishing cosine.
		return f
	}
	g := ggxG2(cosThetaV, cosThetaL, material.ggxAlpha())
	return Clamp(f*g/cosThetaV, 0, 1)
}

// bounceGGX implements the microfacet GGX reflection/transmission model.
// Reflection directions are drawn from the GGX NDF and weighted with Smith
// masking and Fresnel; the remaining energy is sent to diffuse (or to
// refraction for transparent materials). Every returned weight is finite
// and non-negative.
func (i Ray) bounceGGX(info *HitInfo, u, v float64, bounceType BounceType, rnd *rand.Rand) (Ray, bool, float64) {
	n := info.Ray
	material := info.Material
	cosThetaV := safeCos(n.Direction.Dot(i.Direction.Negate()))
	var f float64
	if material.Reflectivity >= 0 {
		f = material.Reflectivity
	} else {
		f = fresnelSchlick(material.f0(), cosThetaV).MaxComponent()
	}
	f = Clamp(f, 0, 1)

	var reflect bool
	switch bounceType {
	case BounceTypeDiffuse:
		reflect = false
	case BounceTypeSpecular:
		reflect = true
	default:
		// Russian roulette over the physically partitioned energy:
		// reflection f, diffuse absorption f*(1-metal), transmission (1-f).
		// Metallic surfaces never take the diffuse ray.
		r := rnd.Float64()
		switch {
		case r < f:
			reflect = true
		case material.Transparent:
			reflect = false
		case r < f+(1-f)*(1-material.Metalness):
			reflect = false
		default:
			// Absorbed by a conductor: no ray, zero weight.
			return Ray{}, false, 0
		}
	}

	if reflect {
		var dir Vector
		if material.Roughness <= 0 {
			// Stable specular limit: exact mirror reflection.
			dir = n.Direction.Reflect(i.Direction)
		} else {
			var ok float64
			dir, _, ok = sampleGGX(n.Direction, i.Direction, material.ggxAlpha(), rnd)
			if ok == 0 || !finiteVector(dir) {
				// Rejected samples exhausted: use the mirror fallback.
				dir = n.Direction.Reflect(i.Direction)
			}
		}
		p := ggxSpecularWeight(material, n.Direction, i.Direction, dir)
		return Ray{n.Origin, dir.Normalize()}, true, p
	}

	if material.Transparent {
		n1, n2 := 1.0, material.Index
		if info.Inside {
			n1, n2 = n2, n1
		}
		refracted := n.Refract(i, n1, n2)
		if refracted.Direction.Length() == 0 {
			// Total internal reflection: energy stays in the reflection lobe.
			tir := Ray{n.Origin, n.Direction.Reflect(i.Direction)}
			tir.Origin = tir.Origin.Add(tir.Direction.MulScalar(1e-4))
			return tir, true, 1
		}
		refracted.Origin = refracted.Origin.Add(refracted.Direction.MulScalar(1e-4))
		return refracted, true, Clamp(1-f, 0, 1)
	}

	// Diffuse: metals have no diffuse component, so the weight carries the
	// (1-metalness) energy factor.
	return n.WeightedBounce(u, v, rnd), false, Clamp((1-f)*(1-material.Metalness), 0, 1)
}

func (i Ray) Bounce(info *HitInfo, u, v float64, bounceType BounceType, rnd *rand.Rand) (Ray, bool, float64) {
	material := info.Material
	if material.ggxEnabled() {
		return i.bounceGGX(info, u, v, bounceType, rnd)
	}
	n := info.Ray
	n1, n2 := 1.0, material.Index
	if info.Inside {
		n1, n2 = n2, n1
	}
	var p float64
	if material.Reflectivity >= 0 {
		p = material.Reflectivity
	} else {
		p = n.Reflectance(i, n1, n2)
	}
	var reflect bool
	switch bounceType {
	case BounceTypeAny:
		reflect = rnd.Float64() < p
	case BounceTypeDiffuse:
		reflect = false
	case BounceTypeSpecular:
		reflect = true
	}
	if reflect {
		reflected := n.Reflect(i)
		return reflected.ConeBounce(material.Gloss, u, v, rnd), true, p
	} else if material.Transparent {
		refracted := n.Refract(i, n1, n2)
		refracted.Origin = refracted.Origin.Add(refracted.Direction.MulScalar(1e-4))
		return refracted.ConeBounce(material.Gloss, u, v, rnd), true, 1 - p
	} else {
		return n.WeightedBounce(u, v, rnd), false, 1 - p
	}
}
