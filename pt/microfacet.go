package pt

import (
	"math"
	"math/rand"
)

// Microfacet GGX (Trowbridge-Reitz) BRDF support.
//
// A Material with Roughness >= 0 opts into the microfacet path:
// reflection directions are sampled from the GGX normal distribution
// function (NDF) and weighted with the Smith GGX masking/shadowing term
// and a Schlick Fresnel reflectance. Materials with Roughness < 0 keep
// the legacy Gloss cone model exactly as before, so existing scenes are
// unaffected.
//
// Roughness is the artist-facing linear roughness parameter in [0, 1];
// alpha = roughness^2 is the shape parameter used by the NDF. A roughness
// of zero degenerates to a stable perfect specular mirror.

// ggxAlphaMin clamps alpha away from zero for numerical stability while
// the roughness == 0 case is handled analytically elsewhere.
const ggxAlphaMin = 1e-6

// finiteOr clamps non-finite values to the supplied fallback.
func finiteOr(x, fallback float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return fallback
	}
	return x
}

// sanitizeMaterial clamps every material parameter into a usable range and
// replaces non-finite values with safe defaults, so sampling never sees
// NaN/Inf inputs and every computed weight stays finite and non-negative.
func sanitizeMaterial(m Material) Material {
	// Roughness: < 0 disables the microfacet path (legacy Gloss model).
	// Non-finite values fall back to a safe mid-range roughness.
	if !math.IsNaN(m.Roughness) && !math.IsInf(m.Roughness, 0) {
		if m.Roughness >= 0 {
			m.Roughness = Clamp(m.Roughness, 0, 1)
		}
	} else {
		m.Roughness = 1
	}
	// Metalness: fully dielectric (0) to fully metallic (1).
	m.Metalness = Clamp(finiteOr(m.Metalness, 0), 0, 1)
	// Refractive index: must be a positive finite number.
	m.Index = finiteOr(m.Index, 1.5)
	if !(m.Index > 1e-3) || math.IsInf(m.Index, 0) {
		m.Index = 1.5
	}
	// Gloss is a cone angle in radians; mirror (0) to hemisphere (pi/2).
	m.Gloss = Clamp(finiteOr(m.Gloss, 0), 0, math.Pi/2)
	// Tint interpolates toward the material color; clamp to [0, 1].
	m.Tint = Clamp(finiteOr(m.Tint, 0), 0, 1)
	// Explicit reflectivity is in [0, 1]; negative means "use Fresnel".
	// Non-finite values revert to the Fresnel-driven default.
	m.Reflectivity = Clamp(finiteOr(m.Reflectivity, -1), -1, 1)
	m.BumpMultiplier = finiteOr(m.BumpMultiplier, 1)
	m.Emittance = math.Max(0, finiteOr(m.Emittance, 0))
	if !finiteColor(m.Color) {
		m.Color = Black
	}
	return m
}

func finiteColor(c Color) bool {
	return !math.IsNaN(c.R) && !math.IsInf(c.R, 0) &&
		!math.IsNaN(c.G) && !math.IsInf(c.G, 0) &&
		!math.IsNaN(c.B) && !math.IsInf(c.B, 0)
}

// ggxEnabled reports whether the material uses the microfacet model.
func (m Material) ggxEnabled() bool {
	return m.Roughness >= 0 && !math.IsNaN(m.Roughness) && !math.IsInf(m.Roughness, 0)
}

// ggxAlpha returns the squared-roughness shape parameter, clamped away from
// zero for the stochastic sampling path (roughness == 0 is handled exactly).
func (m Material) ggxAlpha() float64 {
	a := m.Roughness * m.Roughness
	if a < ggxAlphaMin {
		return ggxAlphaMin
	}
	return a
}

// f0 returns the normal-incidence Fresnel reflectance. Conductors use their
// material color as base reflectance (tinted reflection, no diffuse return);
// dielectrics derive it from the refractive index. Metalness blends the two
// for partially metallic materials.
func (m Material) f0() Color {
	var dielectric float64
	eta := m.Index
	if eta > 1 {
		r := (eta - 1) / (eta + 1)
		dielectric = r * r
	} else {
		// Abnormal indices (0 < eta <= 1): use a neutral 4% baseline
		// instead of producing a negative or inverted reflectance.
		dielectric = 0.04
	}
	dielectric = Clamp(dielectric, 0, 1)
	base := Color{dielectric, dielectric, dielectric}
	return base.Mix(m.Color, m.Metalness)
}

// safeCos clamps a cosine to [0, 1], mapping NaN to 0 (grazing incidence,
// which gives the conservative full-reflectance Fresnel response).
func safeCos(c float64) float64 {
	if math.IsNaN(c) {
		return 0
	}
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

// fresnelSchlick evaluates the Schlick Fresnel approximation for the given
// cosine of the view angle, with the grazing term clamped so the result is
// always finite and in [0, 1].
func fresnelSchlick(f0 Color, cosTheta float64) Color {
	c := safeCos(cosTheta)
	t := 1 - c
	t5 := t * t * t * t * t
	return Color{
		R: f0.R + (1-f0.R)*t5,
		G: f0.G + (1-f0.G)*t5,
		B: f0.B + (1-f0.B)*t5,
	}
}

func fresnelSchlickScalar(f0, cosTheta float64) float64 {
	f0 = Clamp(f0, 0, 1)
	c := safeCos(cosTheta)
	t := 1 - c
	return f0 + (1-f0)*t*t*t*t*t
}

// ggxDistribution evaluates the GGX (Trowbridge-Reitz) normal distribution
// function for microfacet normal m. n and m are assumed unit length and
// cosThetaM = dot(n, m) is clamped to the upper hemisphere.
func ggxDistribution(n, m Vector, alpha float64) float64 {
	cosThetaM := n.Dot(m)
	if cosThetaM <= 0 {
		// back-facing microfacet contributes no energy
		return 0
	}
	cos2 := cosThetaM * cosThetaM
	a2 := alpha * alpha
	d := cos2*(a2-1) + 1
	if d < 1e-12 {
		return 0
	}
	return a2 / (math.Pi * d * d)
}

// ggxG1 is the Smith-GGX monodirectional masking term (Schlick approximation).
// cosTheta is clamped to [0, 1], so grazing and back-facing directions stay
// finite and non-negative.
func ggxG1(cosTheta, alpha float64) float64 {
	cosTheta = safeCos(cosTheta)
	a2 := alpha * alpha
	denom := cosTheta + math.Sqrt(a2+(1-a2)*cosTheta*cosTheta)
	if denom < 1e-12 {
		return 0
	}
	return 2 * cosTheta / denom
}

// ggxG2 is the Smith-GGX bidirectional masking/shadowing term using the
// separable (joint) approximation.
func ggxG2(cosThetaV, cosThetaL, alpha float64) float64 {
	return ggxG1(cosThetaV, alpha) * ggxG1(cosThetaL, alpha)
}

// localFrame builds an orthonormal basis (t, b, n) from a unit normal that is
// stable even when the normal nearly coincides with the helper axis.
func localFrame(n Vector) (Vector, Vector) {
	helper := V(1, 0, 0)
	if math.Abs(n.Dot(helper)) > 0.99 {
		helper = V(0, 1, 0)
	}
	t := helper.Cross(n).Normalize()
	b := n.Cross(t)
	return t, b
}

// sampleGGX samples a microfacet normal m from the GGX NDF in the hemisphere
// around n and returns the reflection direction of the incident ray i along
// with the sampled microfacet normal. Microfacet normals that would produce a
// back-facing reflection are resampled (a bounded number of times) before
// falling back to the mirror direction, so the returned ray is always valid.
func sampleGGX(n, i Vector, alpha float64, rnd *rand.Rand) (Vector, Vector, float64) {
	t, b := localFrame(n)
	for attempt := 0; attempt < 16; attempt++ {
		// GGX NDF sampling: cos(theta_m)^2 = (1-u)/(1+(a^2-1)u)
		u := rnd.Float64()
		v := rnd.Float64()
		a2 := alpha * alpha
		cos2 := (1 - u) / (1 + (a2-1)*u)
		if cos2 < 0 || cos2 > 1 || math.IsNaN(cos2) || math.IsInf(cos2, 0) {
			continue
		}
		cosM := math.Sqrt(cos2)
		sinM := math.Sqrt(math.Max(0, 1-cos2))
		phi := 2 * math.Pi * v
		m := t.MulScalar(sinM * math.Cos(phi)).
			Add(b.MulScalar(sinM * math.Sin(phi))).
			Add(n.MulScalar(cosM))
		if m.Length() < 1e-12 {
			continue
		}
		m = m.Normalize()
		// Reflect the incident direction about the sampled microfacet normal.
		wo := m.Reflect(i)
		if !finiteVector(wo) || wo.Dot(n) <= EPS {
			continue // back-facing microfacet: resample
		}
		return wo.Normalize(), m, 1
	}
	// Exhausted retries: fall back to the geometric mirror reflection.
	return n.Reflect(i), n, 0
}

func finiteVector(v Vector) bool {
	return !math.IsNaN(v.X) && !math.IsInf(v.X, 0) &&
		!math.IsNaN(v.Y) && !math.IsInf(v.Y, 0) &&
		!math.IsNaN(v.Z) && !math.IsInf(v.Z, 0)
}
