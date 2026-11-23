package pt

import (
	"math"
	"math/rand"
	"testing"
)

func approx(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("%s: non-finite result %v (want %v)", name, got, want)
	}
	if math.Abs(got-want) > tol {
		t.Fatalf("%s: got %v, want %v (tol %v)", name, got, want, tol)
	}
}

func TestDistribution1DUniform(t *testing.T) {
	d := NewDistribution1D([]float64{2, 2, 2, 2})
	if !d.Valid() {
		t.Fatal("uniform distribution should be valid")
	}
	// every bin carries 1/4 of the mass, density per unit u = 1
	for _, u := range []float64{0.01, 0.25, 0.5, 0.75, 0.99} {
		_, _, pdf := d.SampleContinuous(u)
		approx(t, "uniform pdf", pdf, 1, 1e-12)
	}
	for _, v := range []float64{0, 0.5, 1} {
		approx(t, "uniform lookup pdf", d.ContinuousPDF(v), 1, 1e-12)
	}
	// samples stay inside [0,1) and are monotonic in the input
	prev := -1.0
	for i := 0; i < 100; i++ {
		u := float64(i) / 100
		value, index, _ := d.SampleContinuous(u)
		if value < 0 || value >= 1 || index < 0 || index >= 4 {
			t.Fatalf("sample out of range: value=%v index=%d", value, index)
		}
		if value < prev {
			t.Fatal("CDF sampling must be monotonic")
		}
		prev = value
	}
}

func TestDistribution1DWeighted(t *testing.T) {
	d := NewDistribution1D([]float64{1, 3})
	if !d.Valid() {
		t.Fatal("weighted distribution should be valid")
	}
	approx(t, "integral", d.Integral(), 4, 1e-12)
	// bin 0 ends at CDF 0.25
	value, index, pdf := d.SampleContinuous(0.1)
	if index != 0 {
		t.Fatalf("u=0.1 must land in bin 0, got %d", index)
	}
	approx(t, "bin0 value", value, 0.2, 1e-12)
	approx(t, "bin0 pdf", pdf, 0.5, 1e-12)
	_, index, pdf = d.SampleContinuous(0.5)
	if index != 1 {
		t.Fatalf("u=0.5 must land in bin 1, got %d", index)
	}
	approx(t, "bin1 pdf", pdf, 1.5, 1e-12)
	approx(t, "bin0 lookup", d.ContinuousPDF(0.2), 0.5, 1e-12)
	approx(t, "bin1 lookup", d.ContinuousPDF(0.8), 1.5, 1e-12)
}

func TestDistribution1DFallbacks(t *testing.T) {
	for name, w := range map[string][]float64{
		"empty": {},
		"zero":  {0, 0, 0},
		"nan":   {math.NaN(), math.NaN()},
		"inf":   {math.Inf(1), math.Inf(1)},
		"neg":   {-1, -2},
		"mixed": {0, math.NaN(), math.Inf(-1), -5},
	} {
		d := NewDistribution1D(w)
		if d.Valid() {
			t.Fatalf("%s weights must produce an invalid distribution", name)
		}
		if v, idx, pdf := d.SampleContinuous(0.5); pdf != 0 || idx != 0 || v != 0 {
			t.Fatalf("%s: invalid distribution must return zero pdf", name)
		}
		if d.ContinuousPDF(0.5) != 0 {
			t.Fatalf("%s: invalid distribution pdf lookup must be zero", name)
		}
	}
	var nilDist *Distribution1D
	if nilDist.Valid() {
		t.Fatal("nil distribution must be invalid")
	}
}

func TestDistribution1DEmpirical(t *testing.T) {
	weights := []float64{1, 2, 5, 2}
	d := NewDistribution1D(weights)
	rnd := rand.New(rand.NewSource(1))
	counts := make([]int, len(weights))
	const n = 200000
	for i := 0; i < n; i++ {
		_, index, _ := d.SampleContinuous(rnd.Float64())
		counts[index]++
	}
	total := 0.0
	for _, w := range weights {
		total += w
	}
	for i, w := range weights {
		want := float64(n) * w / total
		rel := math.Abs(float64(counts[i])-want) / want
		if rel > 0.03 {
			t.Fatalf("bin %d empirical mass off by %.2f%% (got %d, want %.0f)", i, rel*100, counts[i], want)
		}
	}
}

func TestDistribution2DUniform(t *testing.T) {
	w, h := 8, 4
	weights := make([]float64, w*h)
	for i := range weights {
		weights[i] = 1
	}
	d := NewDistribution2D(weights, w, h)
	if !d.Valid() {
		t.Fatal("uniform 2D distribution should be valid")
	}
	// uniform joint density per unit area is 1
	for _, u := range []float64{0.1, 0.5, 0.9} {
		for _, v := range []float64{0.1, 0.5, 0.9} {
			approx(t, "uniform 2D pdf", d.PDF(u, v), 1, 1e-12)
		}
	}
	rnd := rand.New(rand.NewSource(2))
	for i := 0; i < 1000; i++ {
		u, v, pdf := d.Sample(rnd.Float64(), rnd.Float64())
		if u < 0 || u >= 1 || v < 0 || v >= 1 || pdf <= 0 {
			t.Fatalf("bad 2D sample u=%v v=%v pdf=%v", u, v, pdf)
		}
	}
}

func TestDistribution2DFallbacks(t *testing.T) {
	if d := NewDistribution2D(nil, 4, 4); d.Valid() {
		t.Fatal("nil weights must be invalid")
	}
	if d := NewDistribution2D([]float64{1, 2}, 0, 2); d.Valid() {
		t.Fatal("zero width must be invalid")
	}
	zeros := make([]float64, 16)
	if d := NewDistribution2D(zeros, 4, 4); d.Valid() {
		t.Fatal("all-zero weights must be invalid")
	}
	// a single live texel is enough for a valid distribution
	zeros[10] = 1
	d := NewDistribution2D(zeros, 4, 4)
	if !d.Valid() {
		t.Fatal("single live texel must be valid")
	}
	u, v, pdf := d.Sample(0.5, 0.5)
	if pdf <= 0 || u < 0 || v < 0 {
		t.Fatal("single live texel sampling failed")
	}
}
