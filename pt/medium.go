package pt

import "math"

// Participating-media limits. They keep every result finite and bound the
// amount of work performed for pathological inputs (zero/negative step sizes,
// non-finite coefficients, extremely long intervals, ...).
const (
	// defaultVolumeStep matches the historical Volume isosurface march step.
	defaultVolumeStep = 1.0 / 512.0
	// maxVolumeStep bounds the configured march step in world-space t units.
	maxVolumeStep = 1.0
	// maxVolumeCoefficient clamps absorption/scattering coefficients. A
	// non-finite coefficient is first sanitized to zero; +Inf is therefore
	// treated as "no coefficient" rather than an instant overflow.
	maxVolumeCoefficient = 1e6
	// maxVolumeDensity clamps density samples fed to the integrators.
	maxVolumeDensity = 1e6
	// maxVolumeOpticalDepth saturates optical depth, so transmittance is
	// always in [exp(-maxVolumeOpticalDepth), 1] and never becomes negative
	// or zero through floating point underflow arithmetic (it can still
	// collapse to exactly 0.0 via math.Exp, which is a defined result).
	maxVolumeOpticalDepth = 100.0
	// maxVolumeSegments bounds the number of march segments on one ray,
	// giving extremely long intervals a hard cost ceiling.
	maxVolumeSegments = 1 << 20
	// maxAdaptiveRefine limits midpoint refinement inside a voxel segment.
	maxAdaptiveRefine = 4
	// defaultMaxScatterEvents bounds stochastic scattering depth.
	defaultMaxScatterEvents = 3
	maxScatterEventsLimit   = 1 << 16
)

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func sanitizeCoefficient(x float64) float64 {
	if math.IsNaN(x) || x < 0 || math.IsInf(x, -1) {
		return 0
	}
	if math.IsInf(x, 1) || x > maxVolumeCoefficient {
		return maxVolumeCoefficient
	}
	return x
}

func sanitizeMediumColor(c Color) Color {
	return Color{
		R: sanitizeCoefficient(c.R),
		G: sanitizeCoefficient(c.G),
		B: sanitizeCoefficient(c.B),
	}
}

// PhaseFunction describes the angular distribution of scattered radiance at
// a point in a participating medium.
type PhaseFunction interface {
	// Evaluate returns p(in, out) for unit directions. Convention: both
	// vectors point along the flow of radiance; the scattering angle is the
	// angle between them.
	Evaluate(in, out Vector) float64
	// Sample draws an outgoing direction for a given incoming direction and
	// returns it together with the probability density it was drawn with.
	Sample(in Vector, rnd RandomSource) (out Vector, pdf float64)
}

// RandomSource is the subset of *math/rand.Rand used by the integrators.
type RandomSource interface {
	Float64() float64
}

// IsotropicPhase is the 1/(4*pi) phase function.
type IsotropicPhase struct{}

func (IsotropicPhase) Evaluate(in, out Vector) float64 {
	return 1.0 / (4 * math.Pi)
}

func (IsotropicPhase) Sample(in Vector, rnd RandomSource) (Vector, float64) {
	// Uniform sphere sampling via the canonical (cos elevation, azimuth)
	// inversion; uses the caller's RNG so renders stay reproducible.
	z := rnd.Float64()*2 - 1
	r := math.Sqrt(math.Max(0, 1-z*z))
	phi := 2 * math.Pi * rnd.Float64()
	return Vector{r * math.Cos(phi), r * math.Sin(phi), z}, 1.0 / (4 * math.Pi)
}

// HenyeyGreensteinPhase implements the Henyey-Greenstein phase function with
// asymmetry parameter g in (-1, 1): g < 0 back-scattering, g > 0 forward.
type HenyeyGreensteinPhase struct {
	G float64
}

func (p HenyeyGreensteinPhase) cosTheta(u float64) float64 {
	g := p.G
	if math.Abs(g) < 1e-3 {
		return 1 - 2*u
	}
	// Inverse CDF of the HG distribution.
	s := (1 - g*g) / (1 - g + 2*g*u)
	return (1 + g*g - s*s) / (2 * g)
}

func (p HenyeyGreensteinPhase) Evaluate(in, out Vector) float64 {
	g := p.G
	cos := in.Dot(out)
	den := 1 + g*g - 2*g*cos
	return (1 - g*g) / (4 * math.Pi * math.Pow(den, 1.5))
}

func (p HenyeyGreensteinPhase) Sample(in Vector, rnd RandomSource) (Vector, float64) {
	cosTheta := p.cosTheta(rnd.Float64())
	sinTheta := math.Sqrt(math.Max(0, 1-cosTheta*cosTheta))
	phi := 2 * math.Pi * rnd.Float64()
	// Build an orthonormal basis around the incoming direction.
	t := orthonormalVector(in)
	b := in.Cross(t)
	d := Vector{}
	d = d.Add(t.MulScalar(sinTheta * math.Cos(phi)))
	d = d.Add(b.MulScalar(sinTheta * math.Sin(phi)))
	d = d.Add(in.MulScalar(cosTheta))
	out := d.Normalize()
	return out, p.Evaluate(in, out)
}

// Medium configures a heterogeneous participating medium bound to a Volume.
// A nil Medium leaves the Volume behaving as a legacy isosurface shape.
type Medium struct {
	// Step is the nominal ray-march step size (world-space t units).
	// Non-finite or non-positive values fall back to defaultVolumeStep.
	Step float64
	// Absorption and Scattering are the per-channel coefficient spectra
	// sigma_a and sigma_s. Effective coefficients are multiplied by the
	// local density sampled from the volume grid.
	Absorption Color
	Scattering Color
	// Phase samples scattered directions. If nil, an isotropic phase
	// function (or Henyey-Greenstein built from G) is used.
	Phase PhaseFunction
	// G is used only when Phase is nil to build a Henyey-Greenstein phase.
	G float64
	// MaxScatterEvents bounds the number of stochastic scattering events
	// on a single camera/indirect ray. Zero disables scattering; negative
	// values select the default. Absorption always terminates a path.
	MaxScatterEvents int
}

// normalized returns a sanitized copy of the medium with every field
// guaranteed finite and every invariant enforced. It never panics.
func (m *Medium) normalized() *Medium {
	cp := *m
	cp.Absorption = sanitizeMediumColor(cp.Absorption)
	cp.Scattering = sanitizeMediumColor(cp.Scattering)
	if !isFinite(cp.Step) || cp.Step <= 0 {
		cp.Step = defaultVolumeStep
	}
	if cp.Step > maxVolumeStep {
		cp.Step = maxVolumeStep
	}
	if cp.Phase == nil {
		g := cp.G
		if !isFinite(g) {
			g = 0
		}
		g = Clamp(g, -0.999, 0.999)
		if math.Abs(g) < 1e-3 {
			cp.Phase = IsotropicPhase{}
		} else {
			cp.Phase = HenyeyGreensteinPhase{g}
		}
	}
	if cp.MaxScatterEvents < 0 {
		cp.MaxScatterEvents = defaultMaxScatterEvents
	}
	if cp.MaxScatterEvents > maxScatterEventsLimit {
		cp.MaxScatterEvents = maxScatterEventsLimit
	}
	return &cp
}

// extinction returns sigma_t = sigma_a + sigma_s.
func (m *Medium) extinction() Color {
	return m.Absorption.Add(m.Scattering)
}

// orthonormalVector returns a deterministic unit vector perpendicular to d.
func orthonormalVector(d Vector) Vector {
	axis := Vector{0, 1, 0}
	if math.Abs(d.Y) > 0.9 {
		axis = Vector{1, 0, 0}
	}
	return d.Cross(axis).Normalize()
}
