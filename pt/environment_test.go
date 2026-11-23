package pt

import (
	"image"
	"math"
	"math/rand"
	"testing"
)

// ---- fixed in-memory HDRI fixtures -------------------------------------

// newEnvTexture builds a ColorTexture from a pixel callback. Pixels are
// supplied in image order (row 0 = top of the image), matching how
// NewTexture ingests an image.Image.
func newEnvTexture(w, h int, fn func(x, y int) Color) *ColorTexture {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	data := make([]Color, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := fn(x, y)
			data[y*w+x] = c
			im.Set(x, y, c.RGBA())
		}
	}
	return &ColorTexture{Width: w, Height: h, Data: data}
}

// sunSpotHDRI is a deterministic HDR environment: a dim uniform background
// plus a small, very bright circular spot (the "HDRI sun") centered at the
// given image-space pixel. Its luminance is concentrated in a tiny fraction
// of the sphere, exactly the regime where uniform escape-ray sampling
// converges slowly and importance sampling should win.
func sunSpotHDRI(w, h int, cx, cy, radius int, bg, sun float64) *ColorTexture {
	return newEnvTexture(w, h, func(x, y int) Color {
		dx := float64(x - cx)
		dy := float64(y - cy)
		v := bg
		if dx*dx+dy*dy <= float64(radius*radius) {
			v = sun
		}
		return Color{v, v, v}
	})
}

func uniformEnvTexture(w, h int, v float64) *ColorTexture {
	return newEnvTexture(w, h, func(x, y int) Color {
		return Color{v, v, v}
	})
}

// ---- validity and fallback behavior ------------------------------------

func TestEnvDistributionFallbacks(t *testing.T) {
	// nil texture
	if d := NewEnvDistribution(nil, 0); d.Valid() {
		t.Fatal("nil texture must yield an invalid distribution")
	}
	// black texture
	black := uniformEnvTexture(64, 32, 0)
	if d := NewEnvDistribution(black, 0); d.Valid() {
		t.Fatal("black texture must yield an invalid distribution")
	}
	// non-finite pixels only
	bad := newEnvTexture(16, 8, func(x, y int) Color {
		return Color{math.NaN(), math.Inf(1), -1}
	})
	if d := NewEnvDistribution(bad, 0); d.Valid() {
		t.Fatal("fully non-finite texture must yield an invalid distribution")
	}
	// degenerate dimensions
	line := uniformEnvTexture(128, 1, 1)
	if d := NewEnvDistribution(line, 0); !d.Valid() {
		t.Fatal("single-row texture with positive energy must still be valid")
	}
	// a handful of bad pixels among good ones must be tolerated
	mostlyGood := newEnvTexture(32, 16, func(x, y int) Color {
		if x == 3 && y == 4 {
			return Color{math.NaN(), math.Inf(1), -100}
		}
		return Color{0.5, 0.5, 0.5}
	})
	if d := NewEnvDistribution(mostlyGood, 0); !d.Valid() {
		t.Fatal("a texture with isolated bad pixels must be valid")
	}
	// invalid distributions never produce a usable sample or pdf
	invalid := NewEnvDistribution(black, 0)
	rnd := rand.New(rand.NewSource(3))
	if dir, pdf, _, _ := invalid.Sample(rnd); pdf != 0 || dir != (Vector{}) {
		t.Fatal("invalid distribution sample must report zero pdf")
	}
	if invalid.PDF(V(1, 0, 0)) != 0 {
		t.Fatal("invalid distribution pdf lookup must be zero")
	}
}

func TestEnvDistributionMonochrome(t *testing.T) {
	// a single-color texture is proportional to solid angle, so the
	// solid-angle pdf must be the uniform 1/(4*pi) everywhere.
	// Use a fine grid so the piecewise-constant approximation is tight.
	tex := uniformEnvTexture(512, 256, 0.7)
	d := NewEnvDistribution(tex, 0)
	if !d.Valid() {
		t.Fatal("monochrome texture must be valid")
	}
	rnd := rand.New(rand.NewSource(4))
	want := 1 / (4 * math.Pi)
	for i := 0; i < 500; i++ {
		dir := RandomUnitVector(rnd)
		approx(t, "uniform solid-angle pdf", d.PDF(dir), want, want*0.005)
	}
}

func TestEnvDistributionExtremeAspect(t *testing.T) {
	// very wide and very tall textures must not crash or produce NaNs
	for _, dims := range [][2]int{{4096, 8}, {8, 4096}, {1, 1}, {3, 2}} {
		tex := sunSpotHDRI(dims[0], dims[1], dims[0]/2, dims[1]/2, 1, 0.01, 10)
		d := NewEnvDistribution(tex, 0)
		if !d.Valid() {
			t.Fatalf("dims %v: distribution must be valid", dims)
		}
		rnd := rand.New(rand.NewSource(5))
		for i := 0; i < 100; i++ {
			dir, pdf, _, _ := d.Sample(rnd)
			if !isFinite(pdf) || pdf <= 0 || !isFinite(dir.X) {
				t.Fatalf("dims %v: non-finite sample", dims)
			}
			if got := d.PDF(dir); math.Abs(got-pdf) > pdf*1e-9 {
				t.Fatalf("dims %v: sample pdf %v != lookup pdf %v", dims, pdf, got)
			}
		}
	}
}

func TestEnvDistributionPoleConcentration(t *testing.T) {
	// put the bright spot on the north pole rows; polar solid-angle
	// weighting must keep every pdf finite (no division by zero at theta=0)
	tex := newEnvTexture(64, 32, func(x, y int) Color {
		v := 0.01
		if y <= 1 {
			v = 50
		}
		return Color{v, v, v}
	})
	d := NewEnvDistribution(tex, 0)
	if !d.Valid() {
		t.Fatal("polar texture must be valid")
	}
	for _, dir := range []Vector{
		{0, 1, 0}, {0, -1, 0}, // exact poles
		{1e-15, 1, 1e-15}, {-1e-15, 1, 0},
	} {
		pdf := d.PDF(dir.Normalize())
		if !isFinite(pdf) {
			t.Fatalf("pole direction %v produced non-finite pdf %v", dir, pdf)
		}
	}
	rnd := rand.New(rand.NewSource(6))
	for i := 0; i < 1000; i++ {
		_, pdf, _, _ := d.Sample(rnd)
		if !isFinite(pdf) || pdf <= 0 {
			t.Fatalf("polar texture sample has bad pdf %v", pdf)
		}
	}
}

// ---- pdf normalization and consistency ---------------------------------

// checkPDFNormalized verifies two properties of the solid-angle pdf:
//  1. exact integration over every distribution bin sums to 1 (this is
//     immune to point-sampling error and to the bounded pdf at the poles);
//  2. pointwise lookups over random sphere directions are always finite
//     and non-negative (no division by zero, no NaN).
func checkPDFNormalized(t *testing.T, d *EnvDistribution, samples int, seed int64) {
	t.Helper()
	w, h := d.width, d.height
	sum := 0.0
	dv := 1.0 / float64(h)
	for j := 0; j < h; j++ {
		v0 := float64(j) * dv
		vc := v0 + dv/2
		lat0 := v0*math.Pi - math.Pi/2
		lat1 := (v0+dv)*math.Pi - math.Pi/2
		bandSolidAngle := math.Sin(lat1) - math.Sin(lat0)
		for i := 0; i < w; i++ {
			uc := (float64(i) + 0.5) / float64(w)
			pdfW := d.pdfOmega(vc, d.dist.PDF(uc, vc))
			if !isFinite(pdfW) || pdfW < 0 {
				t.Fatalf("non-finite bin pdf %v at (%d,%d)", pdfW, i, j)
			}
			sum += pdfW * bandSolidAngle * 2 * math.Pi / float64(w)
		}
	}
	approx(t, "exact integral of pdf over sphere", sum, 1, 1e-9)
	rnd := rand.New(rand.NewSource(seed))
	for i := 0; i < samples; i++ {
		pdf := d.PDF(RandomUnitVector(rnd))
		if !isFinite(pdf) || pdf < 0 {
			t.Fatalf("non-finite point pdf %v", pdf)
		}
	}
}

func TestEnvDistributionPDFNormalized(t *testing.T) {
	tex := sunSpotHDRI(128, 64, 30, 20, 4, 0.05, 30)
	checkPDFNormalized(t, NewEnvDistribution(tex, 0), 200000, 7)
	checkPDFNormalized(t, NewEnvDistribution(uniformEnvTexture(96, 48, 1), 0), 100000, 8)
	polar := newEnvTexture(64, 32, func(x, y int) Color {
		v := 0.02
		if y >= 30 {
			v = 20
		}
		return Color{v, v, v}
	})
	checkPDFNormalized(t, NewEnvDistribution(polar, 0), 200000, 9)
}

func TestEnvDistributionSamplePDFConsistency(t *testing.T) {
	tex := sunSpotHDRI(128, 64, 90, 40, 6, 0.02, 25)
	d := NewEnvDistribution(tex, 0)
	rnd := rand.New(rand.NewSource(10))
	for i := 0; i < 2000; i++ {
		dir, pdf, _, _ := d.Sample(rnd)
		got := d.PDF(dir)
		if math.Abs(got-pdf) > math.Max(1e-9, pdf*1e-9) {
			t.Fatalf("sample pdf %.6f != lookup pdf %.6f", pdf, got)
		}
	}
}

func TestEnvDistributionTargetsBrightSpot(t *testing.T) {
	// the sampler should spend most of its samples inside the bright spot,
	// even though it covers only a small fraction of the sphere
	const r = 3
	tex := sunSpotHDRI(128, 64, 96, 32, r, 0.01, 100)
	d := NewEnvDistribution(tex, 0)
	rnd := rand.New(rand.NewSource(11))
	hits := 0
	const n = 50000
	for i := 0; i < n; i++ {
		dir, pdf, _, _ := d.Sample(rnd)
		if pdf <= 0 {
			t.Fatal("zero pdf from valid distribution")
		}
		u, v := envDirectionToUV(dir, 0)
		x := u * 128
		// image rows are top-down while v is the latitude fraction
		y := (1 - v) * 64
		dx := x - 96
		dy := y - 32
		if dx*dx+dy*dy <= r*r {
			hits++
		}
	}
	// spot area is roughly 28 px out of 8192 (~0.34%); importance sampling
	// must land there the overwhelming majority of the time
	fraction := float64(hits) / n
	if fraction < 0.8 {
		t.Fatalf("only %.1f%% of samples hit the bright spot (want >= 80%%)", fraction*100)
	}
}

// ---- texture rotation and the seam -------------------------------------

func TestEnvDistributionRotation(t *testing.T) {
	tex := sunSpotHDRI(128, 64, 32, 32, 5, 0.01, 40)
	angle := math.Pi / 3
	d0 := NewEnvDistribution(tex, 0)
	d1 := NewEnvDistribution(tex, angle)
	// Rotation about Y preserves solid angle, so the rotated distribution
	// must remain normalized and its samples must stay pdf-consistent and
	// finite. (A pointwise equality across rotations is not asserted
	// because the piecewise-constant pdf jumps at latitude-row boundaries,
	// and two arbitrary directions can straddle a bin edge.)
	checkPDFNormalized(t, d0, 100000, 12)
	checkPDFNormalized(t, d1, 100000, 15)
	rnd := rand.New(rand.NewSource(14))
	for i := 0; i < 2000; i++ {
		dir, pdf, _, v := d1.Sample(rnd)
		if !isFinite(pdf) || pdf <= 0 || v < 0 || v > 1 {
			t.Fatalf("bad rotated sample pdf=%v v=%v", pdf, v)
		}
		if got := d1.PDF(dir); math.Abs(got-pdf) > pdf*1e-9 {
			t.Fatalf("rotated sample pdf %.6f != lookup %.6f", pdf, got)
		}
	}
	// Inverse mapping: the longitude of a sampled direction, converted back
	// through the same rotation, must round-trip into the sampled u bin.
	for i := 0; i < 2000; i++ {
		dir, _, u, v := d1.Sample(rnd)
		uu, vv := envDirectionToUV(dir, angle)
		uu = Fract(Fract(uu) + 1)
		if math.Abs(vv-v) > 1.0/float64(d1.height) {
			t.Fatalf("latitude round-trip failed: %v vs %v", vv, v)
		}
		if du := math.Abs(uu - u); du > 1.0/float64(d1.width) && du < 1-1.0/float64(d1.width) {
			t.Fatalf("longitude round-trip failed: %v vs %v", uu, u)
		}
	}
}

func TestEnvDistributionSeamAndWrap(t *testing.T) {
	tex := sunSpotHDRI(128, 64, 2, 32, 3, 0.01, 30) // spot straddles u=0
	d := NewEnvDistribution(tex, 0.7)               // non-zero rotation pushes u out of [0,1]
	rnd := rand.New(rand.NewSource(13))
	// directions looking down -X / +X map near the seam under rotation
	for _, dir := range []Vector{{1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1}} {
		pdf := d.PDF(dir)
		if !isFinite(pdf) || pdf < 0 {
			t.Fatalf("seam direction %v gave bad pdf %v", dir, pdf)
		}
	}
	for i := 0; i < 1000; i++ {
		dir, pdf, _, _ := d.Sample(rnd)
		if got := d.PDF(dir); math.Abs(got-pdf) > pdf*1e-9 {
			t.Fatalf("rotated seam pdf mismatch: %.6f vs %.6f", pdf, got)
		}
	}
}
