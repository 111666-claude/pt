package pt

import (
	"math"
)

// isFinite reports whether x is neither NaN nor infinite.
func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// Distribution1D is a reusable piecewise-constant 1D discrete distribution.
// It stores a normalized CDF over len(weights) bins so it can be sampled
// repeatedly without rebuilding. A zero integral produces an invalid
// distribution (Valid() == false) and callers must fall back to another
// sampling strategy instead of dividing by it.
type Distribution1D struct {
	cdf     []float64 // length count+1, cdf[0] == 0, cdf[count] == integral
	funcInt float64   // integral of the (unnormalized) weights
	count   int
}

// NewDistribution1D builds a distribution from non-negative weights.
// Weights that are NaN, infinite or negative are treated as zero.
func NewDistribution1D(weights []float64) *Distribution1D {
	n := len(weights)
	d := &Distribution1D{
		cdf:   make([]float64, n+1),
		count: n,
	}
	if n == 0 {
		return d
	}
	integral := 0.0
	for i, w := range weights {
		if !math.IsInf(w, 0) && !math.IsNaN(w) && w > 0 {
			integral += w
		}
		d.cdf[i+1] = integral
	}
	d.funcInt = integral
	if integral <= 0 {
		// leave the CDF at zero; nothing can be sampled from this
		return d
	}
	scale := 1 / integral
	for i := range d.cdf {
		d.cdf[i] *= scale
	}
	d.cdf[n] = 1 // guard against accumulated floating point drift
	return d
}

// Valid reports whether the distribution has positive mass and can be sampled.
func (d *Distribution1D) Valid() bool {
	return d != nil && d.count > 0 && d.funcInt > 0 && isFinite(d.funcInt)
}

// Count returns the number of bins.
func (d *Distribution1D) Count() int {
	return d.count
}

// Integral returns the unnormalized integral of the supplied weights.
func (d *Distribution1D) Integral() float64 {
	return d.funcInt
}

// SampleContinuous maps a uniform sample in [0,1) to a value in [0,1),
// returning the sampled position, the bin index, and the probability
// density with respect to the [0,1) domain (i.e. probability per unit u).
func (d *Distribution1D) SampleContinuous(u float64) (float64, int, float64) {
	if !d.Valid() {
		return 0, 0, 0
	}
	if u <= 0 {
		u = 0
	} else if u >= 1 {
		u = math.Float64frombits(math.Float64bits(1) - 1)
	}
	// find first bin whose upper CDF value is greater than u
	lo, hi := 0, d.count
	for lo < hi {
		mid := (lo + hi) / 2
		if d.cdf[mid+1] <= u {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	index := lo
	du := 0.0
	if index > 0 {
		du = d.cdf[index]
	}
	binMass := d.cdf[index+1] - du
	if binMass <= 0 {
		binMass = 0
	}
	offset := 0.0
	if binMass > 0 {
		offset = (u - du) / binMass
	}
	value := (float64(index) + offset) / float64(d.count)
	pdf := binMass * float64(d.count) // density per unit u
	return value, index, pdf
}

// ContinuousPDF returns the density per unit u of a value in [0,1).
func (d *Distribution1D) ContinuousPDF(value float64) float64 {
	if !d.Valid() {
		return 0
	}
	if value < 0 || value > 1 {
		return 0
	}
	index := int(value * float64(d.count))
	if index >= d.count {
		index = d.count - 1
	}
	binMass := d.cdf[index+1] - d.cdf[index]
	return binMass * float64(d.count)
}

// Distribution2D is a reusable two-dimensional piecewise-constant
// distribution over a [0,1) x [0,1) parameter domain, built from a
// row-major grid of non-negative weights. Rows are marginalized into a
// 1D distribution and each row keeps its own conditional 1D distribution,
// which is the standard decomposition used for environment map sampling.
type Distribution2D struct {
	rows      []*Distribution1D // conditional distribution within each row
	marginals *Distribution1D   // distribution over rows
	width     int
	height    int
}

// NewDistribution2D builds a 2D distribution from row-major weights of the
// given dimensions.
func NewDistribution2D(weights []float64, width, height int) *Distribution2D {
	d := &Distribution2D{width: width, height: height}
	if width <= 0 || height <= 0 || len(weights) < width*height {
		return d
	}
	d.rows = make([]*Distribution1D, height)
	marginals := make([]float64, height)
	for y := 0; y < height; y++ {
		row := weights[y*width : (y+1)*width : (y+1)*width]
		rd := NewDistribution1D(row)
		d.rows[y] = rd
		marginals[y] = rd.Integral()
	}
	d.marginals = NewDistribution1D(marginals)
	return d
}

// Valid reports whether the distribution carries any probability mass.
func (d *Distribution2D) Valid() bool {
	return d != nil && d.width > 0 && d.height > 0 && d.marginals != nil && d.marginals.Valid()
}

// Width returns the number of columns.
func (d *Distribution2D) Width() int { return d.width }

// Height returns the number of rows.
func (d *Distribution2D) Height() int { return d.height }

// Sample maps two uniform samples in [0,1) to a point (u,v) in [0,1)^2 and
// returns the joint probability density per unit area of the parameter domain.
func (d *Distribution2D) Sample(u0, u1 float64) (u, v, pdf float64) {
	if !d.Valid() {
		return 0, 0, 0
	}
	v, row, pdfV := d.marginals.SampleContinuous(u0)
	rowDist := d.rows[row]
	if !rowDist.Valid() {
		return 0, 0, 0
	}
	u, _, pdfU := rowDist.SampleContinuous(u1)
	return u, v, pdfU * pdfV
}

// PDF returns the joint density per unit area of the parameter domain at (u,v).
func (d *Distribution2D) PDF(u, v float64) float64 {
	if !d.Valid() {
		return 0
	}
	if u < 0 || u > 1 || v < 0 || v > 1 {
		return 0
	}
	row := int(v * float64(d.height))
	if row >= d.height {
		row = d.height - 1
	}
	rowDist := d.rows[row]
	if !rowDist.Valid() {
		return 0
	}
	return d.marginals.ContinuousPDF(v) * rowDist.ContinuousPDF(u)
}
