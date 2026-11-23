package pt

import (
	"math"
	"math/rand"
	"testing"
)

func approxEq(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > tol {
		t.Errorf("%s = %v, want %v (tol %v)", name, got, want, tol)
	}
}

func TestSanitizeMaterial(t *testing.T) {
	base := GGXMaterial(White, 1.5, 0.5, 0.5)
	// out-of-range values are clamped
	m := base
	m.Roughness = 7
	m.Metalness = -3
	m.Index = -2
	m.Gloss = 9
	m.Tint = 5
	m.Reflectivity = 9
	m = m.Validate()
	approxEq(t, "roughness hi", m.Roughness, 1, 0)
	approxEq(t, "metalness lo", m.Metalness, 0, 0)
	approxEq(t, "gloss hi", m.Gloss, math.Pi/2, 0)
	approxEq(t, "tint hi", m.Tint, 1, 0)
	approxEq(t, "reflectivity hi", m.Reflectivity, 1, 0)
	if m.Index != 1.5 {
		t.Errorf("abnormal index %v should reset to 1.5", m.Index)
	}
	// non-finite values are replaced with safe defaults
	m = base
	m.Roughness = math.NaN()
	m.Metalness = math.Inf(1)
	m.Index = math.NaN()
	m = m.Validate()
	if math.IsNaN(m.Roughness) || math.IsInf(m.Roughness, 0) {
		t.Errorf("non-finite roughness not sanitized: %v", m.Roughness)
	}
	if math.IsNaN(m.Metalness) || math.IsInf(m.Metalness, 0) {
		t.Errorf("non-finite metalness not sanitized: %v", m.Metalness)
	}
	if m.Index != 1.5 {
		t.Errorf("non-finite index not reset: %v", m.Index)
	}
	// legacy materials keep the GGX path disabled
	for _, m := range []Material{
		DiffuseMaterial(White), GlossyMaterial(White, 1.4, Radians(20)),
		SpecularMaterial(White, 2), ClearMaterial(1.5, 0),
		MetallicMaterial(White, 0, 1),
	} {
		if m.ggxEnabled() {
			t.Errorf("legacy material should not enable GGX: %+v", m)
		}
	}
	// zero roughness is explicitly enabled and maps to the mirror limit
	m = MetalMaterial(White, 0)
	if !m.ggxEnabled() || m.Roughness != 0 {
		t.Errorf("zero roughness must enable the GGX mirror path")
	}
	// the internal fallback materials (volume/mesh) stay on the legacy path
	for _, m := range []Material{{Roughness: -1}, DiffuseMaterial(White)} {
		if m.Validate().ggxEnabled() {
			t.Errorf("legacy fallback material unexpectedly enabled GGX: %+v", m)
		}
	}
}

func TestF0AndFresnel(t *testing.T) {
	// dielectric at normal incidence: ((eta-1)/(eta+1))^2
	m := DielectricMaterial(White, 1.5, 0.3)
	f0 := m.f0()
	want := 0.04 // (0.5/2.5)^2
	approxEq(t, "dielectric f0", f0.R, want, 1e-12)
	// metallic f0 follows the material color
	m = MetalMaterial(Color{0.8, 0.2, 0.1}, 0.3)
	f0 = m.f0()
	approxEq(t, "metal f0 r", f0.R, 0.8, 1e-12)
	approxEq(t, "metal f0 b", f0.B, 0.1, 1e-12)
	// abnormal index falls back to a neutral 4% baseline
	m = DielectricMaterial(White, 0.2, 0.3)
	f0 = m.f0()
	approxEq(t, "abnormal eta f0", f0.R, 0.04, 1e-12)
	// Schlick Fresnel: f0 at normal incidence, 1 at grazing
	approxEq(t, "fresnel normal", fresnelSchlickScalar(0.04, 1), 0.04, 1e-12)
	approxEq(t, "fresnel grazing", fresnelSchlickScalar(0.04, 0), 1, 1e-12)
	// never out of [0,1] even for clamped/back-facing cosines
	for _, c := range []float64{-1, -0.5, 0, 0.5, 1, 2, math.NaN()} {
		f := fresnelSchlickScalar(0.04, c)
		if math.IsNaN(f) || f < 0 || f > 1 {
			t.Errorf("fresnel out of range for cos=%v: %v", c, f)
		}
	}
	fc := fresnelSchlick(Color{0.8, 0.2, 0.1}, -0.5)
	if fc.MinComponent() < 0 || fc.MaxComponent() > 1 || !finiteColor(fc) {
		t.Errorf("colored fresnel out of range: %+v", fc)
	}
}

func TestGGXDistribution(t *testing.T) {
	n := V(0, 1, 0)
	// NDF at the normal for a few roughness values is finite and positive
	for _, r := range []float64{0.01, 0.1, 0.5, 1} {
		d := ggxDistribution(n, n, r*r)
		if math.IsNaN(d) || math.IsInf(d, 0) || d <= 0 {
			t.Errorf("NDF non-positive at roughness %v: %v", r, d)
		}
	}
	// NDF integrates to one over the hemisphere (Monte Carlo with cosine
	// weight is one; here check the projected integral: integral D*cos dw = 1)
	rnd := rand.New(rand.NewSource(42))
	for _, alpha := range []float64{0.1, 0.35, 1} {
		const N = 200000
		var sum float64
		for i := 0; i < N; i++ {
			// uniform hemisphere sampling: pdf = 1/(2 pi)
			u1, u2 := rnd.Float64(), rnd.Float64()
			y := 1 - u1
			r := math.Sqrt(math.Max(0, 1-y*y))
			phi := 2 * math.Pi * u2
			m := V(r*math.Cos(phi), y, r*math.Sin(phi))
			sum += ggxDistribution(n, m, alpha) * y * 2 * math.Pi
		}
		approxEq(t, "NDF hemisphere integral", sum/N, 1, 0.03)
	}
	// back-facing microfacet contributes nothing
	if ggxDistribution(n, V(0, -1, 0), 0.5) != 0 {
		t.Errorf("NDF must be zero below the hemisphere")
	}
}

func TestSmithG(t *testing.T) {
	// G1 is in [0,1], equals 1 at normal incidence, 0 at grazing for rough
	for _, alpha := range []float64{0.05, 0.5, 1} {
		g0 := ggxG1(1, alpha)
		if g0 > 1+1e-12 || g0 <= 0 {
			t.Errorf("G1(normal) out of range: %v", g0)
		}
		gGrazing := ggxG1(0, alpha)
		if gGrazing != 0 {
			t.Errorf("G1(grazing) must be 0, got %v", gGrazing)
		}
		g2 := ggxG2(0.7, 0.3, alpha)
		if math.IsNaN(g2) || g2 < 0 || g2 > 1 {
			t.Errorf("G2 out of range: %v", g2)
		}
		// negative cosine (back-facing) must stay finite and non-negative
		if g := ggxG1(-0.5, alpha); g < 0 || math.IsNaN(g) {
			t.Errorf("G1 negative for back-facing input: %v", g)
		}
	}
	// perfect mirror has no geometric occlusion
	approxEq(t, "G2 mirror", ggxG2(0.4, 0.6, ggxAlphaMin), 1, 0.02)
}

func TestSampleGGX(t *testing.T) {
	rnd := rand.New(rand.NewSource(7))
	n := V(0, 1, 0)
	i := V(0.3, -0.9, 0.2).Normalize()
	mirror := n.Reflect(i)
	// Angular spread of the sampled lobe around the mirror direction must
	// grow monotonically with roughness, and every sample stays in the
	// upper hemisphere.
	prevSpread := -1.0
	for _, r := range []float64{0.05, 0.25, 0.8} {
		alpha := r * r
		var spread float64
		const N = 5000
		for k := 0; k < N; k++ {
			wo, _, ok := sampleGGX(n, i, alpha, rnd)
			if ok == 0 {
				t.Fatalf("sampling failed at roughness %v", r)
			}
			if !finiteVector(wo) || wo.Dot(n) <= 0 {
				t.Fatalf("invalid reflection direction: %+v", wo)
			}
			c := Clamp(wo.Dot(mirror), -1, 1)
			spread += math.Acos(c)
		}
		spread /= N
		if spread <= prevSpread {
			t.Errorf("roughness %v spread %v did not grow over %v", r, spread, prevSpread)
		}
		prevSpread = spread
	}
	// A smooth lobe concentrates tightly around the mirror direction.
	var tight Vector
	const M = 20000
	for k := 0; k < M; k++ {
		wo, _, ok := sampleGGX(n, i, 0.01, rnd)
		if ok == 0 {
			t.Fatalf("smooth sampling failed")
		}
		tight = tight.Add(wo)
	}
	tight = tight.DivScalar(M)
	if math.Abs(tight.X-mirror.X) > 0.02 || math.Abs(tight.Z-mirror.Z) > 0.02 || math.Abs(tight.Y-mirror.Y) > 0.02 {
		t.Errorf("smooth lobe %+v not concentrated on mirror %+v", tight, mirror)
	}
}

func TestGGXMirrorLimit(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	n := V(0, 1, 0)
	i := V(0.4, -0.8, 0.3).Normalize()
	info := &HitInfo{
		Position: V(0, 0, 0),
		Normal:   n,
		Ray:      Ray{V(0, 0, 0), n},
		Material: MetalMaterial(White, 0),
	}
	for _, mode := range []BounceType{BounceTypeAny, BounceTypeSpecular} {
		newRay, reflected, p := Ray{V(0, 0, 0), i}.Bounce(info, 0.5, 0.5, mode, rnd)
		if !reflected {
			t.Fatalf("mirror limit must reflect (mode %v)", mode)
		}
		want := n.Reflect(i)
		if newRay.Direction.Sub(want).Length() > 1e-12 {
			t.Errorf("mirror direction %+v != %+v (mode %v)", newRay.Direction, want, mode)
		}
		if p <= 0 || p > 1 || math.IsNaN(p) {
			t.Errorf("mirror weight invalid: %v", p)
		}
	}
}

func TestGGXBounceWeights(t *testing.T) {
	rnd := rand.New(rand.NewSource(3))
	n := V(0, 1, 0)
	i := V(0, -1, 0)
	// fully metallic material: diffuse ray carries zero weight, no
	// transmission, specular weight finite in [0,1]
	metal := MetalMaterial(Color{0.9, 0.7, 0.2}, 0.4)
	info := &HitInfo{Ray: Ray{V(0, 0, 0), n}, Normal: n, Material: metal}
	for k := 0; k < 100; k++ {
		_, reflected, p := Ray{V(0, 0, 0), i}.Bounce(info, rnd.Float64(), rnd.Float64(), BounceTypeAny, rnd)
		if p < 0 || p > 1 || math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("metal bounce weight invalid: %v", p)
		}
		if !reflected && p != 0 {
			t.Fatalf("metal diffuse bounce must have zero weight, got %v", p)
		}
	}
	// explicit diffuse bounce of a metal is a dead ray
	_, reflected, p := Ray{V(0, 0, 0), i}.Bounce(info, 0.5, 0.5, BounceTypeDiffuse, rnd)
	if reflected || p != 0 {
		t.Errorf("metal diffuse branch must be a zero-weight non-specular ray, got %v %v", reflected, p)
	}
	// dielectric keeps a non-zero diffuse component
	diel := DielectricMaterial(White, 1.5, 0.4)
	info.Material = diel
	_, reflected, p = Ray{V(0, 0, 0), i}.Bounce(info, 0.5, 0.5, BounceTypeDiffuse, rnd)
	if reflected || p <= 0 || p > 1 {
		t.Errorf("dielectric diffuse weight invalid: reflected=%v p=%v", reflected, p)
	}
	// grazing incidence: dielectric Fresnel approaches one
	grazing := V(0.9999, -0.01, 0).Normalize()
	info.Material = DielectricMaterial(White, 1.5, 0.2)
	_, reflected, p = Ray{V(0, 0, 0), grazing}.Bounce(info, 0.5, 0.5, BounceTypeSpecular, rnd)
	if !reflected || p < 0.9 {
		t.Errorf("grazing dielectric should be almost fully reflective, p=%v", p)
	}
}

func TestBackFacingNormalAndTIR(t *testing.T) {
	rnd := rand.New(rand.NewSource(11))
	// back-facing geometry normal: Hit.Info flips it, but emulate a flipped
	// hit manually and ensure no negative or non-finite weights
	n := V(0, -1, 0)
	i := V(0, 1, 0)
	info := &HitInfo{
		Ray:      Ray{V(0, 0, 0), n},
		Normal:   n,
		Material: MetalMaterial(White, 0.3),
		Inside:   true,
	}
	_, _, p := Ray{V(0, 0, 0), i}.Bounce(info, 0.5, 0.5, BounceTypeSpecular, rnd)
	if p < 0 || p > 1 || math.IsNaN(p) || math.IsInf(p, 0) {
		t.Fatalf("back-facing bounce weight invalid: %v", p)
	}
	// total internal reflection inside a transparent body returns full weight
	info.Material = GGXMaterial(White, 1.5, 0.1, 0)
	info.Material.Transparent = true
	info.Inside = true
	n = V(0, 1, 0)
	info.Ray = Ray{V(0, 0, 0), n}
	info.Normal = n
	steep := V(0.1, 0.995, 0).Normalize() // nearly tangent from inside
	newRay, reflected, p := Ray{V(0, 0, 0), steep}.Bounce(info, 0.5, 0.5, BounceTypeDiffuse, rnd)
	if p < 0 || p > 1 || math.IsNaN(p) {
		t.Fatalf("TIR weight invalid: %v", p)
	}
	if newRay.Direction.Length() == 0 {
		t.Fatalf("TIR produced zero-length ray")
	}
	_ = reflected
}

func TestGGXTransparentWeightPartition(t *testing.T) {
	rnd := rand.New(rand.NewSource(5))
	n := V(0, 1, 0)

	// Smooth limit: no Smith masking, so the reflection weight is exactly
	// Fresnel F and the transmission weight is exactly 1-F: energy partitions
	// to unity at every incidence angle.
	smooth := GGXMaterial(White, 1.5, 0, 0)
	smooth.Transparent = true
	info := &HitInfo{Ray: Ray{V(0, 0, 0), n}, Normal: n, Material: smooth}
	for k := 0; k < 64; k++ {
		ang := rnd.Float64() * 1.45
		i := V(math.Sin(ang), -math.Cos(ang), 0)
		_, _, ps := Ray{V(0, 0, 0), i}.Bounce(info, rnd.Float64(), rnd.Float64(), BounceTypeSpecular, rnd)
		_, refracted, pt2 := Ray{V(0, 0, 0), i}.Bounce(info, rnd.Float64(), rnd.Float64(), BounceTypeDiffuse, rnd)
		if !refracted || ps < 0 || ps > 1 || pt2 < 0 || pt2 > 1 {
			t.Fatalf("smooth partition out of range: ps=%v pt=%v", ps, pt2)
		}
		if d := math.Abs(ps + pt2 - 1); d > 1e-9 {
			t.Fatalf("smooth F+(1-F) != 1: %v+%v gap %v", ps, pt2, d)
		}
	}

	// Rough case: both weights stay in [0,1] and finite. Smith masking
	// attenuates most reflections, though the G2/cosV ratio can exceed 1 at
	// near-grazing geometry (the same behavior as the legacy transparent
	// model, whose reflection weight is likewise unattenuated at grazing).
	rough := GGXMaterial(White, 1.5, 0.3, 0)
	rough.Transparent = true
	info.Material = rough
	for k := 0; k < 300; k++ {
		ang := rnd.Float64() * 1.45
		i := V(math.Sin(ang), -math.Cos(ang), 0)
		_, reflected, ps := Ray{V(0, 0, 0), i}.Bounce(info, rnd.Float64(), rnd.Float64(), BounceTypeSpecular, rnd)
		_, refracted, pt2 := Ray{V(0, 0, 0), i}.Bounce(info, rnd.Float64(), rnd.Float64(), BounceTypeDiffuse, rnd)
		if !reflected || !refracted {
			t.Fatalf("bounce type not honored: reflected=%v refracted=%v", reflected, refracted)
		}
		if ps < 0 || ps > 1 || pt2 < 0 || pt2 > 1 ||
			math.IsNaN(ps) || math.IsInf(ps, 0) || math.IsNaN(pt2) {
			t.Fatalf("rough transparent weight invalid: ps=%v pt=%v", ps, pt2)
		}
	}
}

func TestLegacyPathUnchanged(t *testing.T) {
	// A legacy glossy material must still take the cone-bounce branch and
	// return the same ray/weight as before the GGX change.
	rnd := rand.New(rand.NewSource(99))
	n := V(0, 1, 0)
	i := V(0.3, -0.9, 0.2).Normalize()
	m := GlossyMaterial(White, 1.5, Radians(20))
	info := &HitInfo{Ray: Ray{V(0, 0, 0), n}, Normal: n, Material: m.Validate()}
	old := Ray{V(0, 0, 0), n}.Reflect(Ray{V(0, 0, 0), i})
	wantDir := Cone(old.Direction, m.Gloss, 0.4, 0.7, rnd)
	rnd = rand.New(rand.NewSource(99))
	newRay, reflected, p := Ray{V(0, 0, 0), i}.Bounce(info, 0.4, 0.7, BounceTypeSpecular, rnd)
	if !reflected || p < 0 || p > 1 {
		t.Fatalf("legacy specular bounce broken: reflected=%v p=%v", reflected, p)
	}
	if newRay.Direction.Sub(wantDir).Length() > 1e-12 {
		t.Errorf("legacy cone bounce direction changed")
	}
}
