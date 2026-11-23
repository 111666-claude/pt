package pt

import (
	"math"
	"math/rand"
	"sort"
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
	return &DefaultSampler{firstHitSamples, maxBounces, true, true, LightModeRandom, SpecularModeNaive, -1}
}

func NewDirectSampler() *DefaultSampler {
	return &DefaultSampler{1, 0, true, false, LightModeAll, SpecularModeAll, -1}
}

type DefaultSampler struct {
	FirstHitSamples int
	MaxBounces      int
	DirectLighting  bool
	SoftShadows     bool
	LightMode       LightMode
	SpecularMode    SpecularMode
	// ScatterEvents caps the number of stochastic scattering events
	// evaluated along camera and indirect rays. -1 uses each medium's own
	// MaxScatterEvents limit; 0 disables stochastic scattering so only the
	// deterministic uncollided transmittance and direct lighting remain.
	ScatterEvents int
}

func (s *DefaultSampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	return s.sample(scene, ray, true, s.FirstHitSamples, 0, rnd)
}

func (s *DefaultSampler) sample(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	if depth > s.MaxBounces {
		return Black
	}
	// Participating media: uncollided transmittance (exact up to the march
	// discretization), stochastic in-scattering with next-event estimation,
	// and transmission of whatever surface the ray finally reaches. All ray
	// classes enter here, so shadow/reflection/refraction rays integrate the
	// same density field.
	if scene.HasMedia() {
		return s.sampleMedia(scene, ray, emission, samples, depth, rnd)
	}
	return s.sampleSurface(scene, ray, emission, samples, depth, rnd)
}

// sampleSurface is the original surface-only transport routine, unchanged in
// behavior for scenes that contain no participating media.
func (s *DefaultSampler) sampleSurface(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	hit := scene.Intersect(ray)
	if !hit.Ok() {
		return s.sampleEnvironment(scene, ray)
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

// ---------------------------------------------------------------------------
// Participating media
// ---------------------------------------------------------------------------

func (s *DefaultSampler) eventLimit(m *Medium) int {
	if s.ScatterEvents >= 0 {
		return s.ScatterEvents
	}
	return m.MaxScatterEvents
}

// sampleMedia evaluates radiance along a ray through a scene containing
// participating media. It returns:
//   - uncollided transmittance times surface/environment radiance, and
//   - stochastic in-scattered radiance via weighted delta tracking with
//     next-event estimation (one random walk per ray).
func (s *DefaultSampler) sampleMedia(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	// Uncollided term: transmittance to the first surface hit (or the
	// environment), then the surface's own radiance.
	hit := scene.Intersect(ray)
	var tLimit float64
	var uncollided Color
	if hit.Ok() {
		tLimit = hit.T
		info := hit.Info(ray)
		material := info.Material
		trans := scene.Transmittance(ray, 0, tLimit)
		if material.Emittance > 0 {
			if s.DirectLighting && !emission {
				uncollided = Black
			} else {
				uncollided = trans.Mul(material.Color.MulScalar(material.Emittance * float64(samples)))
			}
		} else {
			// Transmitted surface response: reuse the surface routine for
			// the BRDF/bounce model, dimming it by beam transmittance.
			uncollided = trans.Mul(s.sampleSurface(scene, ray, emission, samples, depth, rnd))
		}
	} else {
		tLimit = INF
		trans := scene.Transmittance(ray, 0, tLimit)
		uncollided = trans.Mul(s.sampleEnvironment(scene, ray))
	}

	// Stochastic in-scattering: weighted delta tracking over every medium
	// intersected by the ray, one random walk, with next-event estimation.
	scattered := s.deltaTrack(scene, ray, depth, 0, tLimit, rnd)
	return uncollided.Add(scattered)
}

// deltaTrack performs one weighted delta-tracking random walk over the media
// intersected by ray up to tLimit. It returns in-scattered radiance weighted
// by the path throughput. The walk is unbiased: voxel-corner majorants bound
// the true density, null collisions only re-scale the path weight,
// absorption terminates the path, and scattering continues with the phase
// function.
func (s *DefaultSampler) deltaTrack(scene *Scene, ray Ray, depth, events int, tLimit float64, rnd *rand.Rand) Color {
	weight := White
	// A single random walk across every medium on the ray. Each entry is a
	// clipped [lo, hi] interval of one volume; they are visited in t order
	// so the path throughput carries across media boundaries.
	var spans []mediumSpan
	for _, v := range scene.media {
		m := v.mediumConfig()
		if m == nil {
			continue
		}
		b0, b1 := v.Box.Intersect(ray)
		if !isFinite(b0) || !isFinite(b1) || b1 <= b0 || b1 <= 0 {
			continue
		}
		lo := math.Max(b0, 0)
		hi := math.Min(b1, tLimit)
		if hi <= lo {
			continue
		}
		spans = append(spans, mediumSpan{lo, hi, v})
	}
	sortSpans(spans)
	for _, sp := range spans {
		v := sp.vol
		m := v.mediumConfig()
		sigmaT := m.extinction()
		tt := sp.lo
		segments := 0
		// Remaining free-flight distance of the current exponential sample.
		// It is drawn once per medium and carried across voxel boundaries
		// (the exponential distribution is memoryless): re-drawing on every
		// boundary would bias the collision distribution. A new sample is
		// drawn after each null collision and at each medium boundary, where
		// the majorant coefficient changes.
		remaining := math.Inf(1)
		for tt < sp.hi {
			if segments >= maxVolumeSegments {
				break
			}
			segments++
			// Grid coordinate frame for boundary clipping inside this voxel.
			gox, goy, goz := v.worldToGrid(ray.Origin)
			gdx, gdy, gdz := v.worldToGrid(ray.Origin.Add(ray.Direction))
			gdx -= gox
			gdy -= goy
			gdz -= goz
			gx := gox + gdx*tt
			gy := goy + gdy*tt
			gz := goz + gdz*tt
			i, j, k := int(math.Floor(gx)), int(math.Floor(gy)), int(math.Floor(gz))
			majorant := v.voxelMajorant(i, j, k)
			voxelEnd := v.clipToVoxelBoundary(gox, goy, goz, gdx, gdy, gdz, tt, sp.hi)
			if voxelEnd <= tt || !isFinite(voxelEnd) {
				voxelEnd = math.Min(tt+m.Step, sp.hi)
			}
			if majorant <= 0 {
				// Vacuum voxel: jump straight to its far boundary without
				// consuming any free-flight distance.
				tt = voxelEnd
				continue
			}
			majColor := sigmaT.MulScalar(majorant)
			// A single scalar majorant (max channel) drives the free-flight
			// distribution; per-channel weights below stay spectrally exact.
			sigmaMaj := majColor.MaxComponent()
			if sigmaMaj <= 0 || !isFinite(sigmaMaj) {
				tt = voxelEnd
				continue
			}
			if math.IsInf(remaining, 1) {
				r := rnd.Float64()
				if r <= 0 {
					break
				}
				remaining = -math.Log(r) / sigmaMaj
			}
			voxelLen := voxelEnd - tt
			if remaining >= voxelLen {
				// The flight survives this whole voxel: subtract its length
				// and continue into the next voxel without re-drawing.
				remaining -= voxelLen
				tt = voxelEnd
				continue
			}
			tEvent := tt + remaining
			// Collision at tEvent; a fresh free-flight sample is needed if
			// it turns out to be a null collision and the walk continues.
			remaining = math.Inf(1)
			p := ray.Position(tEvent)
			density := v.Density(p)
			sigmaEff := sigmaT.MulScalar(density)
			rho := Color{
				R: safeDiv(sigmaEff.R, sigmaMaj),
				G: safeDiv(sigmaEff.G, sigmaMaj),
				B: safeDiv(sigmaEff.B, sigmaMaj),
			}
			weight = weight.Mul(rho)
			if weight.MaxComponent() <= 0 {
				return Black // fully absorbed through weighting
			}
			// Scalar absorption/extinction ratio for the real/null decision.
			scatterRate := m.Scattering.MulScalar(density).MaxComponent()
			totalRate := sigmaEff.MaxComponent()
			if totalRate <= 0 {
				tt = tEvent
				continue
			}
			if rnd.Float64() > scatterRate/totalRate {
				return Black // absorption: path terminates
			}
			// Real scattering event.
			events++
			if events > s.eventLimit(m) {
				return Black
			}
			out, _ := m.Phase.Sample(ray.Direction, rnd)
			newRay := Ray{p, out}
			result := Black
			if s.DirectLighting {
				// Next-event estimation at the scatter point: weight each
				// light by the phase function of the incident direction.
				result = result.Add(s.sampleLightsAtPhase(scene, p, ray.Direction, m.Phase, rnd))
			}
			result = result.Add(s.sample(scene, newRay, false, 1, depth+1, rnd))
			return weight.Mul(result)
		}
	}
	return Black
}

// SingleScatter integrates exactly one order of in-scattered direct light
// deterministically with the trapezoidal rule over adaptively refined
// segments. It contains no random numbers, so it serves as a numerical
// reference and produces noise-free single-scattering images. Path
// transmittance and light shadow rays both go through the same
// Scene.Transmittance density field; the phase function is applied per light.
func (s *DefaultSampler) SingleScatter(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	if !s.DirectLighting || len(scene.media) == 0 {
		return Black
	}
	result := Black
	for _, v := range scene.media {
		m := v.mediumConfig()
		if m == nil {
			continue
		}
		b0, b1 := v.Box.Intersect(ray)
		if !isFinite(b0) || !isFinite(b1) || b1 <= b0 || b1 <= 0 {
			continue
		}
		sigmaS := m.Scattering
		v.marchSegments(ray, math.Max(b0, 0), b1, func(seg segment) {
			tm := (seg.t0 + seg.t1) / 2
			density := (seg.d0 + seg.d1) / 2
			if density <= 0 {
				return
			}
			dt := seg.t1 - seg.t0
			p := ray.Position(tm)
			trToPoint := scene.Transmittance(ray, 0, tm)
			direct := s.sampleLightsAtPhase(scene, p, ray.Direction, m.Phase, rnd)
			result = result.Add(trToPoint.Mul(direct).Mul(sigmaS.MulScalar(density * dt)))
		})
	}
	return result
}

func safeDiv(a, b float64) float64 {
	if b <= 0 || !isFinite(b) {
		return 0
	}
	return math.Min(1, math.Max(0, a/b))
}

// mediumSpan is a clipped [lo, hi] interval of one volume along a ray.
type mediumSpan struct {
	lo, hi float64
	vol    *Volume
}

func sortSpans(spans []mediumSpan) {
	sort.Slice(spans, func(i, j int) bool { return spans[i].lo < spans[j].lo })
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

// lightGeometry resolves the emissive sphere/box used to aim a shadow ray.
func lightGeometry(light Shape) (center Vector, radius float64) {
	switch t := light.(type) {
	case *Sphere:
		return t.Center, t.Radius
	default:
		box := light.BoundingBox()
		return box.Center(), box.OuterRadius()
	}
}

// sampleLightsAt returns direct-light radiance arriving at an arbitrary
// point p (used for next-event estimation inside a medium). It applies no
// surface cosine term and dims the result by beam transmittance along the
// shadow ray, so participating media attenuate every light.
func (s *DefaultSampler) sampleLightsAt(scene *Scene, p Vector, rnd *rand.Rand) Color {
	return s.sampleLightsAtPhase(scene, p, Vector{}, nil, rnd)
}

// sampleLightsAtPhase is sampleLightsAt with an optional phase function: when
// phase is non-nil each light's contribution is weighted by p(in, toLight).
func (s *DefaultSampler) sampleLightsAtPhase(scene *Scene, p Vector, in Vector, phase PhaseFunction, rnd *rand.Rand) Color {
	nLights := len(scene.Lights)
	if nLights == 0 {
		return Black
	}
	emit := func(light Shape) Color {
		center, radius := lightGeometry(light)
		point := center
		if s.SoftShadows {
			for {
				x := rnd.Float64()*2 - 1
				y := rnd.Float64()*2 - 1
				if x*x+y*y <= 1 {
					l := center.Sub(p).Normalize()
					u := l.Cross(RandomUnitVector(rnd)).Normalize()
					vv := l.Cross(u)
					point = Vector{}
					point = point.Add(u.MulScalar(x * radius))
					point = point.Add(vv.MulScalar(y * radius))
					point = point.Add(center)
					break
				}
			}
		}
		toLight := point.Sub(p)
		dist := toLight.Length()
		if dist <= EPS {
			return Black
		}
		dir := toLight.DivScalar(dist)
		ray := Ray{p, dir}
		hit := scene.Intersect(ray)
		if !hit.Ok() || hit.Shape != light {
			return Black
		}
		trans := scene.Transmittance(ray, 0, dist)
		material := MaterialAt(light, point)
		// Geometric weight for a point receiver aiming at an emissive
		// sphere: its projected solid angle is approximately pi*r^2/d^2.
		// There is no surface cosine term inside a medium, so this differs
		// from the surface sampleLight convention, which folds cosine into
		// its coverage proxy. A point at or inside the light sees full
		// emission.
		geom := 1.0
		if dist > radius {
			geom = math.Min(1, math.Pi*radius*radius/(dist*dist))
		}
		weight := material.Emittance * geom
		c := trans.Mul(material.Color.MulScalar(weight))
		if phase != nil {
			c = c.MulScalar(phase.Evaluate(in, dir))
		}
		return c
	}
	if s.LightMode == LightModeAll {
		var result Color
		for _, light := range scene.Lights {
			result = result.Add(emit(light))
		}
		return result
	}
	return emit(scene.Lights[rand.Intn(nLights)]).MulScalar(float64(nLights))
}

func (s *DefaultSampler) sampleLight(scene *Scene, n Ray, rnd *rand.Rand, light Shape) Color {
	center, radius := lightGeometry(light)

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

	// Participating media attenuate the shadow ray over the full distance
	// to the light. With no media this multiplies by White and is a no-op.
	trans := scene.Transmittance(ray, 0, hyp)

	// get material properties from light
	material := MaterialAt(light, point)

	// combine factors
	m := material.Emittance * diffuse * coverage
	return trans.Mul(material.Color.MulScalar(m))
}
