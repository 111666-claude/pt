package pt

import (
	"bytes"
	"image"
	"image/png"
	"math"
	"testing"
)

const epsilon = 1e-12

// ---- independent reference implementations ----

func refGamma(x, exposure float64) float64 {
	x = x * exposure
	if x < 0 {
		x = 0
	}
	v := math.Pow(x, 1/2.2)
	// the original pipeline clamps when encoding to 16-bit channels
	if v > 1 {
		v = 1
	}
	return v
}

func refReinhard(x, exposure float64) float64 {
	x = x * exposure
	if x < 0 {
		x = 0
	}
	return math.Pow(x/(1+x), 1/2.2)
}

// Hill ACES fit, the curve used by three.js ACESFilmicToneMapping.
// Reference written in the direct form; the implementation uses an
// algebraically identical overflow-free form.
func refACES(x, exposure float64) float64 {
	x = x * exposure
	if x <= 0 {
		return 0
	}
	const a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
	v := (x * (a*x + b)) / (x*(c*x+d) + e)
	if v < 0 {
		v = 0
	} else if v > 1 {
		v = 1
	}
	return math.Pow(v, 1/2.2)
}

func refMapping(t ToneMapping, x float64) float64 {
	switch t.Mapper {
	case ReinhardToneMapper:
		return refReinhard(x, t.Exposure)
	case ACESToneMapper:
		return refACES(x, t.Exposure)
	default:
		return refGamma(x, t.Exposure)
	}
}

// staircase: fixed brightness steps spanning deep shadows, mid-tones and
// blown-out highlights, plus a bright highlight point.
func staircaseValues() []float64 {
	return []float64{
		0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.18, 0.25,
		0.5, 0.75, 1, 1.5, 2, 4, 8, 16, 32, 64, 128, 1000,
	}
}

func near(a, b float64) bool {
	d := math.Abs(a - b)
	return d <= epsilon || d <= 1e-12*math.Max(1, math.Abs(b))
}

// refACESStable mirrors the implementation's overflow-free form, used to
// check inputs large enough to make the direct rational overflow.
func refACESStable(x float64) float64 {
	if x <= 0 {
		return 0
	}
	const a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
	inv := 1 / x
	v := (a + b*inv) / (c + d*inv + e*inv*inv)
	if v < 0 {
		v = 0
	} else if v > 1 {
		v = 1
	}
	return math.Pow(v, 1/2.2)
}

// ---- curve: per-pixel agreement with the reference implementation ----

func TestToneMappingCurvePerPixel(t *testing.T) {
	exposures := []float64{0.5, 1, 2, 4}
	mappings := []ToneMapping{
		{GammaToneMapper, 1},
		{ReinhardToneMapper, 1},
		{ACESToneMapper, 1},
	}
	for _, base := range mappings {
		for _, exposure := range exposures {
			m := ToneMapping{base.Mapper, exposure}
			for _, x := range staircaseValues() {
				got := m.mapChannel(x)
				want := refMapping(m, x)
				if !near(got, want) {
					t.Errorf("%v x=%g: got %.12f want %.12f", m, x, got, want)
				}
				if math.IsNaN(got) || math.IsInf(got, 0) || got < 0 || got > 1 {
					t.Errorf("%v x=%g: output not in [0,1]: %g", m, x, got)
				}
			}
		}
	}
}

// ---- original pt output is preserved byte-for-byte by default ----

func TestDefaultToneMappingMatchesOriginalGamma(t *testing.T) {
	buf := NewBuffer(3, 2)
	values := staircaseValues()
	i := 0
	for y := 0; y < buf.H; y++ {
		for x := 0; x < buf.W; x++ {
			c := Color{values[i%len(values)], values[(i+3)%len(values)], values[(i+7)%len(values)]}
			buf.AddSample(x, y, c)
			i++
		}
	}
	legacy := legacyImage(buf)
	current := buf.Image(ColorChannel)
	defaulted := buf.ImageWithToneMapping(ColorChannel, ToneMapping{})
	if !imagesEqual(legacy, current) {
		t.Fatal("Image(ColorChannel) differs from original gamma pipeline")
	}
	if !imagesEqual(legacy, defaulted) {
		t.Fatal("zero-value ToneMapping does not reproduce the original gamma pipeline")
	}
	// Renderer must default to the same configuration.
	r := NewRenderer(&Scene{}, &Camera{}, NewDirectSampler(), 1, 1)
	if r.ToneMapping != NewToneMapping() {
		t.Fatalf("renderer default tone mapping = %v, want %v", r.ToneMapping, NewToneMapping())
	}
}

// legacyImage reproduces the exact pre-change Buffer.Image implementation.
func legacyImage(b *Buffer) image.Image {
	result := image.NewRGBA64(image.Rect(0, 0, b.W, b.H))
	for y := 0; y < b.H; y++ {
		for x := 0; x < b.W; x++ {
			c := b.Pixels[y*b.W+x].Color().Pow(1 / 2.2)
			result.SetRGBA64(x, y, c.RGBA64())
		}
	}
	return result
}

func imagesEqual(a, b image.Image) bool {
	bounds := a.Bounds()
	if b.Bounds() != bounds {
		return false
	}
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if a.At(x, y) != b.At(x, y) {
				return false
			}
		}
	}
	return true
}

// ---- hue: channel ratios are preserved, no cross-channel mixing ----

func TestToneMappingPreservesHue(t *testing.T) {
	colors := []Color{
		{1, 0.25, 0.0625},
		{0.5, 1, 0.1},
		{0.02, 0.08, 0.4},
		{4, 8, 2},
	}
	mappings := []ToneMapping{
		{GammaToneMapper, 1},
		{ReinhardToneMapper, 1},
		{ACESToneMapper, 1},
		{ReinhardToneMapper, 2},
		{ACESToneMapper, 0.5},
	}
	for _, m := range mappings {
		for _, c := range colors {
			got := m.apply(c)
			want := Color{refMapping(m, c.R), refMapping(m, c.G), refMapping(m, c.B)}
			if !near(got.R, want.R) || !near(got.G, want.G) || !near(got.B, want.B) {
				t.Errorf("%v %v: got %v want %v", m, c, got, want)
			}
			// Each channel is an independent function of the same channel:
			// equal inputs must produce equal outputs (gray stays gray).
			gray := m.apply(Color{0.3, 0.3, 0.3})
			if gray.R != gray.G || gray.G != gray.B {
				t.Errorf("%v: neutral gray shifted in hue: %v", m, gray)
			}
		}
	}
}

// ---- extreme luminance ----

func TestToneMappingExtremes(t *testing.T) {
	mappings := []ToneMapping{
		{GammaToneMapper, 1},
		{ReinhardToneMapper, 1},
		{ACESToneMapper, 1},
	}
	for _, m := range mappings {
		if got := m.mapChannel(0); got != 0 {
			t.Errorf("%v: black mapped to %g, want 0", m, got)
		}
		if got := m.mapChannel(math.Inf(1)); got != 1 {
			t.Errorf("%v: +Inf mapped to %g, want 1", m, got)
		}
		if got := m.mapChannel(math.NaN()); got != 0 {
			t.Errorf("%v: NaN mapped to %g, want 0", m, got)
		}
		if got := m.mapChannel(-100); got != 0 {
			t.Errorf("%v: negative mapped to %g, want 0", m, got)
		}
		// finite but huge values stay finite and within range; the ACES
		// expression must not overflow for the largest representable floats
		for _, huge := range []float64{1e18, 1e100, 1e300, math.MaxFloat64} {
			got := m.mapChannel(huge)
			if math.IsNaN(got) || math.IsInf(got, 0) || got < 0 || got > 1 {
				t.Errorf("%v: %g produced unsafe value %g", m, huge, got)
			}
		}
		// the smallest positive float must stay a valid, near-black value
		if tiny := m.mapChannel(math.SmallestNonzeroFloat64); math.IsNaN(tiny) || tiny < 0 || tiny > 1 {
			t.Errorf("%v: smallest float produced unsafe value %g", m, tiny)
		}
	}
	// Gamma clamps super-white to 1 after encoding instead of overflowing.
	if got := NewToneMapping().mapChannel(1e18); got != 1 {
		t.Errorf("gamma 1e18 = %g, want 1", got)
	}
}

// ---- non-finite and non-positive parameters are rejected ----

func TestInvalidToneMappingParameters(t *testing.T) {
	bad := []ToneMapping{
		{GammaToneMapper, 0},
		{GammaToneMapper, -1},
		{GammaToneMapper, math.NaN()},
		{GammaToneMapper, math.Inf(1)},
		{GammaToneMapper, math.Inf(-1)},
		{ToneMapper(99), 1},
		{ToneMapper(-1), math.NaN()},
	}
	good := NewToneMapping()
	for _, m := range bad {
		if n := m.normalized(); n != good {
			t.Errorf("%v normalized to %v, want %v", m, n, good)
		}
		// Even a malformed configuration must produce safe output.
		got := m.mapChannel(0.5)
		if math.IsNaN(got) || math.IsInf(got, 0) || got < 0 || got > 1 {
			t.Errorf("%v produced unsafe output %g", m, got)
		}
	}
}

// ---- full black, transparent/no-hit and NaN-laden buffers stay finite ----

func TestSpecialBuffersProduceSafePNG(t *testing.T) {
	mappings := []ToneMapping{
		NewToneMapping(),
		{ReinhardToneMapper, 1},
		{ACESToneMapper, 1},
		{ACESToneMapper, 2.5},
	}
	for _, m := range mappings {
		// fully black buffer: the no-hit / transparent background case
		black := NewBuffer(2, 2)
		im := black.ImageWithToneMapping(ColorChannel, m)
		assertFiniteImage(t, im, "black buffer "+m.Mapper.String())

		// buffer whose accumulated mean contains NaN and Inf
		poisoned := NewBuffer(2, 1)
		poisoned.Pixels[0].M = Color{math.NaN(), math.Inf(1), math.Inf(-1)}
		poisoned.Pixels[0].Samples = 1
		poisoned.Pixels[1].M = Color{-5, 0, 1e300}
		poisoned.Pixels[1].Samples = 1
		im = poisoned.ImageWithToneMapping(ColorChannel, m)
		assertFiniteImage(t, im, "poisoned buffer "+m.Mapper.String())

		var buf bytes.Buffer
		if err := png.Encode(&buf, im); err != nil {
			t.Fatalf("png encode failed for %v: %v", m, err)
		}
	}
}

func assertFiniteImage(t *testing.T, im image.Image, label string) {
	t.Helper()
	bounds := im.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := im.At(x, y).RGBA()
			// color.Color values are unsigned integers, so they are finite
			// by construction; verify they stay within the 16-bit range and
			// the alpha is fully opaque.
			if r > 65535 || g > 65535 || b > 65535 || a != 65535 {
				t.Errorf("%s at (%d,%d) out of range: %d %d %d %d", label, x, y, r, g, b, a)
			}
		}
	}
}

// ---- repeated Image calls never mutate the buffer ----

func TestRepeatedImageDoesNotMutateBuffer(t *testing.T) {
	buf := NewBuffer(4, 3)
	for y := 0; y < buf.H; y++ {
		for x := 0; x < buf.W; x++ {
			buf.AddSample(x, y, Color{float64(x) * 1.7, float64(y) * 0.3, 2.5})
		}
	}
	snapshot := make([]Pixel, len(buf.Pixels))
	copy(snapshot, buf.Pixels)

	mappings := []ToneMapping{
		NewToneMapping(),
		{ReinhardToneMapper, 2},
		{ACESToneMapper, 0.5},
	}
	var first image.Image
	for i, m := range mappings {
		for repeat := 0; repeat < 3; repeat++ {
			im := buf.ImageWithToneMapping(ColorChannel, m)
			if i == 0 && repeat == 0 {
				first = im
			} else if i == 0 && !imagesEqual(first, im) {
				t.Fatal("repeated calls returned different images")
			}
		}
		// other channels must keep working alongside tone mapping
		buf.ImageWithToneMapping(VarianceChannel, m)
		buf.ImageWithToneMapping(StandardDeviationChannel, m)
		buf.ImageWithToneMapping(SamplesChannel, m)
	}
	for i, p := range buf.Pixels {
		if p != snapshot[i] {
			t.Fatalf("pixel %d mutated: %+v -> %+v", i, snapshot[i], p)
		}
	}
}

// ---- channel boundaries: mapping applies only to ColorChannel ----

func TestToneMappingOnlyAffectsColorChannel(t *testing.T) {
	buf := NewBuffer(2, 2)
	buf.AddSample(0, 0, Color{4, 2, 1})
	buf.AddSample(0, 0, Color{1, 3, 1})
	buf.AddSample(1, 0, Color{8, 8, 8})
	buf.AddSample(0, 1, Color{0.1, 0.2, 0.4})
	plain := NewToneMapping()
	mapped := ToneMapping{ACESToneMapper, 3}

	colorA := buf.ImageWithToneMapping(ColorChannel, plain)
	colorB := buf.ImageWithToneMapping(ColorChannel, mapped)
	if imagesEqual(colorA, colorB) {
		t.Fatal("ACES mapping had no effect on ColorChannel")
	}
	for _, ch := range []Channel{VarianceChannel, StandardDeviationChannel, SamplesChannel} {
		a := buf.ImageWithToneMapping(ch, plain)
		b := buf.ImageWithToneMapping(ch, mapped)
		if !imagesEqual(a, b) {
			t.Errorf("channel %d changed under tone mapping", ch)
		}
		c := buf.Image(ch)
		if !imagesEqual(a, c) {
			t.Errorf("channel %d no longer matches Buffer.Image output", ch)
		}
	}
}

// ---- SamplesChannel with no samples: no 0/0 NaN ----

func TestEmptySamplesChannel(t *testing.T) {
	buf := NewBuffer(2, 2)
	im := buf.Image(SamplesChannel).(*image.RGBA64)
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			if got := im.RGBA64At(x, y); got.R != 0 || got.G != 0 || got.B != 0 {
				t.Fatalf("empty samples channel produced %v", got)
			}
		}
	}
}

// ---- RGBA64 hardening ----

func TestRGBA64SanitizesNonFinite(t *testing.T) {
	// Any non-finite channel (NaN, +Inf, -Inf) is encoded as 0; the
	// mapping layer is what turns an HDR +Inf into a white display pixel.
	cases := map[Color][3]uint16{
		{math.NaN(), 1, 1}:   {0, 65535, 65535},
		{math.Inf(1), 1, 1}:  {0, 65535, 65535},
		{math.Inf(-1), 1, 1}: {0, 65535, 65535},
		{-1, 2, 0.5}:         {0, 65535, 32767},
		{1e6, 1e-300, 0}:     {65535, 0, 0},
	}
	for c, want := range cases {
		got := c.RGBA64()
		if got.R != want[0] || got.G != want[1] || got.B != want[2] || got.A != 65535 {
			t.Errorf("%v RGBA64 = %v, want %v", c, got, want)
		}
	}
}

// ---- ACES: direct reference vs overflow-free implementation ----

func TestACESStableFormParity(t *testing.T) {
	m := ToneMapping{ACESToneMapper, 1}
	// over the normal HDR range the two algebraic forms must agree
	for i := 1; i <= 100000; i++ {
		x := float64(i) / 10
		got := m.mapChannel(x)
		want := refACES(x, 1)
		if d := math.Abs(got - want); d > 1e-12 {
			t.Fatalf("x=%g: stable form %g differs from direct ref %g by %g", x, got, want, d)
		}
	}
	// extreme inputs use the stable reference instead of the overflowing one
	for _, x := range []float64{1e100, 1e300, math.MaxFloat64} {
		got := m.mapChannel(x)
		want := refACESStable(x)
		if !near(got, want) {
			t.Errorf("x=%g: got %g want %g", x, got, want)
		}
	}
	// the curve asymptote is below 1 before gamma, so output stays valid
	if got := m.mapChannel(math.MaxFloat64); got <= 0 || got > 1 {
		t.Errorf("MaxFloat64 mapped outside [0,1]: %g", got)
	}
}

// ---- monotonicity and highlight rolloff ----

func TestToneMappingMonotonic(t *testing.T) {
	mappings := []ToneMapping{
		{GammaToneMapper, 1},
		{ReinhardToneMapper, 1},
		{ACESToneMapper, 1},
	}
	for _, m := range mappings {
		prev := 0.0
		for i := 1; i <= 5000; i++ {
			x := float64(i) / 10
			got := m.mapChannel(x)
			if got < prev {
				t.Fatalf("%v not monotonic at x=%g: %g < %g", m, x, got, prev)
			}
			prev = got
		}
		// highlights roll off toward 1 without exceeding it
		if m.mapChannel(1000) < 0.999 {
			t.Errorf("%v highlight at x=1000 only %g", m, m.mapChannel(1000))
		}
	}
}

// ---- end-to-end: a real render stays finite through PNG output ----

func TestRenderEndToEndWithToneMapping(t *testing.T) {
	scene := Scene{}
	material := DiffuseMaterial(White)
	scene.Add(NewPlane(V(0, 0, 0), V(0, 0, 1), material))
	scene.Add(NewSphere(V(0, 0, 1), 1, material))
	scene.Add(NewSphere(V(0, 0, 5), 1, LightMaterial(White, 8)))
	camera := LookAt(V(3, 3, 3), V(0, 0, 0.5), V(0, 0, 1), 50)

	for _, m := range []ToneMapping{
		NewToneMapping(),
		{ReinhardToneMapper, 1},
		{ACESToneMapper, 1},
	} {
		renderer := NewRenderer(&scene, &camera, NewSampler(2, 2), 16, 9)
		renderer.SamplesPerPixel = 8
		renderer.NumCPU = 2
		renderer.Verbose = false
		renderer.ToneMapping = m
		im := renderer.Render()
		assertFiniteImage(t, im, "render "+m.Mapper.String())
		var buf bytes.Buffer
		if err := png.Encode(&buf, im); err != nil {
			t.Fatalf("png encode failed: %v", err)
		}
		// accumulation data must remain linear and untouched
		if c := renderer.Buffer.Color(0, 0); math.IsNaN(c.MaxComponent()) || math.IsInf(c.MaxComponent(), 0) {
			t.Fatalf("%v rendered non-finite buffer color %v", m, c)
		}
	}
}

// ---- exposure scales the linear input before the curve ----

func TestExposureEffect(t *testing.T) {
	x := 0.25
	base := ToneMapping{ReinhardToneMapper, 1}.mapChannel(x)
	up := ToneMapping{ReinhardToneMapper, 2}.mapChannel(x)
	down := ToneMapping{ReinhardToneMapper, 0.5}.mapChannel(x)
	if !(down < base && base < up) {
		t.Fatalf("exposure ordering wrong: %g < %g < %g", down, base, up)
	}
	// exposure 2 on x equals exposure 1 on 2x
	if !near(up, ToneMapping{ReinhardToneMapper, 1}.mapChannel(2*x)) {
		t.Fatal("exposure is not applied as a linear pre-multiply")
	}
}
