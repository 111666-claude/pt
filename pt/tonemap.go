package pt

import "math"

// ToneMapper selects the tone mapping curve applied to the ColorChannel
// after linear HDR accumulation.
type ToneMapper int

const (
	// GammaToneMapper reproduces the original pt output: linear values are
	// gamma encoded with an exponent of 1/2.2 and no tone mapping curve.
	GammaToneMapper ToneMapper = iota
	// ReinhardToneMapper applies the simple Reinhard curve x/(1+x).
	ReinhardToneMapper
	// ACESToneMapper applies the ACES filmic approximation (Hill fit,
	// as used by three.js's ACESFilmicToneMapping).
	ACESToneMapper
)

// String returns the name of the tone mapping curve.
func (m ToneMapper) String() string {
	switch m {
	case ReinhardToneMapper:
		return "Reinhard"
	case ACESToneMapper:
		return "ACES"
	default:
		return "Gamma"
	}
}

// ToneMapping configures how linear HDR pixel values are converted to
// display values for the ColorChannel.
//
// Exposure is a linear multiplier applied before the mapping curve; it must
// be a finite, positive number or the value is replaced with the default (1).
// The zero value is normalized to the gamma defaults, reproducing the
// original pt output; use NewToneMapping to request them explicitly.
type ToneMapping struct {
	Mapper   ToneMapper
	Exposure float64
}

// NewToneMapping returns the default tone mapping configuration: gamma-only
// output with an exposure of 1, which is byte-for-byte equivalent to the
// original pt ColorChannel output for finite, non-negative pixel values.
func NewToneMapping() ToneMapping {
	return ToneMapping{GammaToneMapper, 1}
}

// normalized returns a safe copy of the configuration. Non-finite or
// non-positive exposures fall back to the default so that NaN, Inf and
// negative exposures can never reach the image. Unknown mappers fall back
// to gamma.
func (t ToneMapping) normalized() ToneMapping {
	exposure := t.Exposure
	if math.IsNaN(exposure) || math.IsInf(exposure, 0) || exposure <= 0 {
		exposure = 1
	}
	mapper := t.Mapper
	switch mapper {
	case GammaToneMapper, ReinhardToneMapper, ACESToneMapper:
	default:
		mapper = GammaToneMapper
	}
	return ToneMapping{mapper, exposure}
}

// acesFilm is the ACES filmic approximation used by three.js's
// ACESFilmicToneMapping (the Hill RRTAndODTFit expressed with Narkowicz's
// constants). Unlike the literal Narkowicz rational, the numerator here has
// no constant term, so the curve passes through the origin: a black pixel
// stays black and the curve is monotonic.
//
// The expression is divided through by x*x so it contains no multiplication
// by x; the direct form overflows (x*x) for finite but very large inputs.
// The two forms are algebraically identical and agree to within ~1e-15.
func acesFilm(x float64) float64 {
	if x <= 0 {
		return 0
	}
	const a = 2.51
	const b = 0.03
	const c = 2.43
	const d = 0.59
	const e = 0.14
	inv := 1 / x
	return (a + b*inv) / (c + d*inv + e*inv*inv)
}

// mapChannel maps a single linear color channel to a gamma-encoded display
// value in [0, 1]. NaN and Inf inputs are sanitized here instead of being
// propagated through math.Pow into the PNG.
func (t ToneMapping) mapChannel(x float64) float64 {
	t = t.normalized()
	if math.IsNaN(x) {
		return 0
	}
	// Exposure is guaranteed finite and positive, so a +Inf channel stays
	// +Inf rather than turning into NaN.
	x = x * t.Exposure
	if math.IsInf(x, 1) {
		return 1
	}
	if x < 0 {
		x = 0
	}
	var v float64
	switch t.Mapper {
	case ReinhardToneMapper:
		// x is finite and non-negative here.
		v = x / (1 + x)
	case ACESToneMapper:
		v = Clamp(acesFilm(x), 0, 1)
	default:
		// GammaToneMapper: no mapping curve, retain the original behavior.
		v = x
	}
	if v < 0 {
		v = 0
	}
	// Gamma encode (exponent 1/2.2) to match the original pt pipeline.
	v = math.Pow(v, 1/2.2)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return Clamp(v, 0, 1)
}

// apply maps every channel of a linear HDR color.
func (t ToneMapping) apply(c Color) Color {
	return Color{
		t.mapChannel(c.R),
		t.mapChannel(c.G),
		t.mapChannel(c.B),
	}
}
