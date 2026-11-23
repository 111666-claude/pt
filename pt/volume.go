package pt

import (
	"image"
	"math"
)

type Volume struct {
	W, H, D int
	ZScale  float64
	Data    []float64
	Windows []VolumeWindow
	Box     Box
	// Medium turns the Volume into a heterogeneous participating medium.
	// When nil (the default) the volume keeps its historical behavior of
	// being intersected as an isosurface and is completely unchanged.
	Medium *Medium
	// normalized medium, populated by Compile
	medium *Medium
}

type VolumeWindow struct {
	Lo, Hi   float64
	Material Material
}

func NewVolume(box Box, images []image.Image, sliceSpacing float64, windows []VolumeWindow) *Volume {
	w := images[0].Bounds().Size().X
	h := images[0].Bounds().Size().Y
	d := len(images)
	// TODO: w/h aspect ratio
	zs := (sliceSpacing * float64(d)) / float64(w)
	data := make([]float64, w*h*d)
	for z, im := range images {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				r, _, _, _ := im.At(x, y).RGBA()
				f := float64(r) / 65535
				data[x+y*w+z*w*h] = f
			}
		}
	}
	return &Volume{W: w, H: h, D: d, ZScale: zs, Data: data, Windows: windows, Box: box}
}

// NewVolumeData builds a Volume directly from a scalar grid (W*H*D values in
// x-major order). It is the programmatic counterpart of NewVolume and is
// used for analytically defined density fields and tests. A nil or empty
// data slice yields an empty volume for which all sampling returns zero.
func NewVolumeData(box Box, w, h, d int, data []float64, medium *Medium) *Volume {
	if w <= 0 || h <= 0 || d <= 0 || len(data) < w*h*d {
		v := &Volume{W: 0, H: 0, D: 0, ZScale: 1, Data: nil, Box: box, Medium: medium}
		return v
	}
	cp := make([]float64, w*h*d)
	copy(cp, data)
	return &Volume{W: w, H: h, D: d, ZScale: 1, Data: cp, Box: box, Medium: medium}
}

func (v *Volume) Get(x, y, z int) float64 {
	if x < 0 || y < 0 || z < 0 || x >= v.W || y >= v.H || z >= v.D {
		return 0
	}
	return v.Data[x+y*v.W+z*v.W*v.H]
}

// Sample performs trilinear interpolation of the raw grid value using the
// historical [-1, 1] -> full-grid mapping. Out-of-bounds reads resolve to
// zero. Non-finite inputs return zero rather than NaN.
func (v *Volume) Sample(x, y, z float64) float64 {
	if v.W == 0 || v.H == 0 || v.D == 0 {
		return 0
	}
	if !isFinite(x) || !isFinite(y) || !isFinite(z) {
		return 0
	}
	z /= v.ZScale
	x = ((x + 1) / 2) * float64(v.W)
	y = ((y + 1) / 2) * float64(v.H)
	z = ((z + 1) / 2) * float64(v.D)
	return v.sampleGrid(x, y, z)
}

// Density returns the clamped non-negative density at a world-space point.
// Negative grid values are treated as vacuum, out-of-bounds points read zero
// and the result is bounded by maxVolumeDensity.
func (v *Volume) Density(p Vector) float64 {
	if v.W == 0 || v.H == 0 || v.D == 0 {
		return 0
	}
	if !isFinite(p.X) || !isFinite(p.Y) || !isFinite(p.Z) {
		return 0
	}
	gx, gy, gz := v.worldToGrid(p)
	return v.densityGrid(gx, gy, gz)
}

func (v *Volume) worldToGrid(p Vector) (gx, gy, gz float64) {
	gz = p.Z / v.ZScale
	gx = ((p.X + 1) / 2) * float64(v.W)
	gy = ((p.Y + 1) / 2) * float64(v.H)
	gz = ((gz + 1) / 2) * float64(v.D)
	return
}

func (v *Volume) densityGrid(x, y, z float64) float64 {
	d := v.sampleGrid(x, y, z)
	if !isFinite(d) || d < 0 {
		return 0
	}
	if d > maxVolumeDensity {
		return maxVolumeDensity
	}
	return d
}

func (v *Volume) sampleGrid(x, y, z float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	z0 := int(math.Floor(z))
	x1 := x0 + 1
	y1 := y0 + 1
	z1 := z0 + 1
	v000 := v.Get(x0, y0, z0)
	v001 := v.Get(x0, y0, z1)
	v010 := v.Get(x0, y1, z0)
	v011 := v.Get(x0, y1, z1)
	v100 := v.Get(x1, y0, z0)
	v101 := v.Get(x1, y0, z1)
	v110 := v.Get(x1, y1, z0)
	v111 := v.Get(x1, y1, z1)
	x -= float64(x0)
	y -= float64(y0)
	z -= float64(z0)
	c00 := v000*(1-x) + v100*x
	c01 := v001*(1-x) + v101*x
	c10 := v010*(1-y) + v110*y
	c11 := v011*(1-y) + v111*y
	c0 := c00*(1-y) + c10*y
	c1 := c01*(1-y) + c11*y
	c := c0*(1-z) + c1*z
	return c
}

// IsMedium reports whether the volume participates in volumetric transport.
func (v *Volume) IsMedium() bool {
	return v.Medium != nil
}

func (v *Volume) Compile() {
	if v.Medium != nil {
		v.medium = v.Medium.normalized()
	} else {
		v.medium = nil
	}
}

func (v *Volume) mediumConfig() *Medium {
	if v.medium == nil && v.Medium != nil {
		v.medium = v.Medium.normalized()
	}
	return v.medium
}

func (v *Volume) BoundingBox() Box {
	return v.Box
}

func (v *Volume) Sign(a Vector) int {
	s := v.Sample(a.X, a.Y, a.Z)
	for i, window := range v.Windows {
		if s < window.Lo {
			return i + 1
		}
		if s > window.Hi {
			continue
		}
		return 0
	}
	return len(v.Windows) + 1
}

// Intersect retains the historical isosurface behavior exactly. A volume
// configured as a participating medium is never treated as a surface: all
// of its transport happens through density integration.
func (v *Volume) Intersect(ray Ray) Hit {
	if v.IsMedium() {
		return NoHit
	}
	tmin, tmax := v.Box.Intersect(ray)
	step := 1.0 / 512
	start := math.Max(step, tmin)
	sign := -1
	for t := start; t <= tmax; t += step {
		p := ray.Position(t)
		s := v.Sign(p)
		if s == 0 || (sign >= 0 && s != sign) {
			t -= step
			step /= 64
			t += step
			for i := 0; i < 64; i++ {
				if v.Sign(ray.Position(t)) == 0 {
					return Hit{v, t - step, nil}
				}
				t += step
			}
		}
		sign = s
	}
	return NoHit
}

func (v *Volume) UV(p Vector) Vector {
	return Vector{} // not implemented
}

func (v *Volume) MaterialAt(p Vector) Material {
	be := 1e9
	bm := Material{}
	s := v.Sample(p.X, p.Y, p.Z)
	for _, window := range v.Windows {
		if s >= window.Lo && s <= window.Hi {
			return window.Material
		}
		e := math.Min(math.Abs(s-window.Lo), math.Abs(s-window.Hi))
		if e < be {
			be = e
			bm = window.Material
		}
	}
	return bm
}

func (v *Volume) NormalAt(p Vector) Vector {
	eps := 0.001
	n := Vector{
		v.Sample(p.X-eps, p.Y, p.Z) - v.Sample(p.X+eps, p.Y, p.Z),
		v.Sample(p.X, p.Y-eps, p.Z) - v.Sample(p.X, p.Y+eps, p.Z),
		v.Sample(p.X, p.Y, p.Z-eps) - v.Sample(p.X, p.Y, p.Z+eps),
	}
	return n.Normalize()
}

// ---------------------------------------------------------------------------
// Participating-media integration
// ---------------------------------------------------------------------------

// segment describes one adaptive ray-march interval [t0, t1] inside the
// volume grid. d0 and d1 are the clamped densities at the endpoints.
type segment struct {
	t0, t1 float64
	d0, d1 float64
}

// voxelMajorant returns a guaranteed upper bound of the density inside voxel
// (i, j, k): the maximum of the eight corner samples. Trilinear interpolation
// is bounded by its corners, so this majorant is exact for the interpolated
// field and keeps stochastic delta tracking unbiased.
func (v *Volume) voxelMajorant(i, j, k int) float64 {
	m := v.Get(i, j, k)
	for _, c := range [][3]int{
		{i + 1, j, k}, {i, j + 1, k}, {i + 1, j + 1, k},
		{i, j, k + 1}, {i + 1, j, k + 1}, {i, j + 1, k + 1},
		{i + 1, j + 1, k + 1},
	} {
		if x := v.Get(c[0], c[1], c[2]); x > m {
			m = x
		}
	}
	if m < 0 || !isFinite(m) {
		return 0
	}
	if m > maxVolumeDensity {
		return maxVolumeDensity
	}
	return m
}

// refine recursively splits a segment at its midpoint while the midpoint
// density deviates from the linear interpolation of the endpoints. This
// guarantees a narrow peak narrower than one voxel cannot be skipped even
// when marching with a large configured step.
func (v *Volume) refine(ray Ray, t0, t1, d0, d1 float64, depth int, fn func(segment)) {
	if depth >= maxAdaptiveRefine {
		fn(segment{t0, t1, d0, d1})
		return
	}
	tm := (t0 + t1) / 2
	dm := v.Density(ray.Position(tm))
	linear := (d0 + d1) / 2
	if math.Abs(dm-linear) > 1e-9 && depth < maxAdaptiveRefine && t1-t0 > 1e-12 {
		v.refine(ray, t0, tm, d0, dm, depth+1, fn)
		v.refine(ray, tm, t1, dm, d1, depth+1, fn)
		return
	}
	fn(segment{t0, t1, d0, d1})
}

// marchSegments walks the ray through the portion [tmin, tmax] of the volume
// that lies inside its bounding box, clipping each step at voxel boundaries
// (so no narrow peak inside a single voxel can be straddled) and adaptively
// refining segments containing interior peaks. The callback receives every
// finalized segment in t order. The total segment count is capped.
func (v *Volume) marchSegments(ray Ray, tmin, tmax float64, fn func(segment)) {
	if v.W == 0 || v.H == 0 || v.D == 0 {
		return
	}
	m := v.mediumConfig()
	if m == nil {
		return
	}
	// Sanitize and bound the integration interval.
	if !isFinite(tmin) || !isFinite(tmax) || tmax <= tmin {
		return
	}
	if tmin < 0 {
		tmin = 0
	}
	// Clip to the volume box, discarding rays that miss it.
	b0, b1 := v.Box.Intersect(ray)
	if !isFinite(b0) || !isFinite(b1) || b1 <= b0 || b1 <= 0 {
		return
	}
	if b0 > tmin {
		tmin = b0
	}
	if b1 < tmax {
		tmax = b1
	}
	if tmax <= tmin {
		return
	}
	step := m.Step
	if !isFinite(step) || step <= 0 {
		step = defaultVolumeStep
	}
	count := 0
	t := tmin
	// Grid coordinate of the ray origin; used to find the voxel containing
	// any world-space t.
	gox, goy, goz := v.worldToGrid(ray.Origin)
	gdx, gdy, gdz := v.worldToGrid(ray.Origin.Add(ray.Direction))
	gdx -= gox
	gdy -= goy
	gdz -= goz
	for t < tmax {
		if count >= maxVolumeSegments {
			return
		}
		tEnd := t + step
		if tEnd > tmax || !isFinite(tEnd) {
			tEnd = tmax
		}
		// Clip tEnd to the first voxel boundary crossing inside (t, tEnd].
		tEnd = v.clipToVoxelBoundary(gox, goy, goz, gdx, gdy, gdz, t, tEnd)
		if tEnd <= t || !isFinite(tEnd) {
			// Degenerate axis; advance by the configured step instead of
			// stalling on a zero-length segment.
			tEnd = math.Min(t+step, tmax)
			if tEnd <= t {
				return
			}
		}
		d0 := v.Density(ray.Position(t))
		d1 := v.Density(ray.Position(tEnd))
		v.refine(ray, t, tEnd, d0, d1, 0, fn)
		count++
		t = tEnd
	}
}

// clipToVoxelBoundary returns the smallest t in (t, tEnd] at which the ray
// crosses an integer grid coordinate of any axis. Grid coordinates are
// affine in t because a ray is linear, so crossings are computed directly
// without an iterative search.
func (v *Volume) clipToVoxelBoundary(gox, goy, goz, gdx, gdy, gdz, t, tEnd float64) float64 {
	next := tEnd
	clipAxis := func(o, d float64) {
		if math.Abs(d) < 1e-15 {
			return
		}
		// Current grid position and the voxel boundary ahead in sign(d).
		gt := o + d*t
		idx := math.Floor(gt)
		var boundary float64
		if d > 0 {
			boundary = idx + 1
		} else {
			boundary = idx
		}
		tb := (boundary - o) / d
		if tb > t+1e-12 && tb < next {
			next = tb
		}
	}
	clipAxis(gox, gdx)
	clipAxis(goy, gdy)
	clipAxis(goz, gdz)
	return next
}

// OpticalDepth integrates sigma_t * density over [tmin, tmax] using the
// trapezoidal rule over adaptively refined segments. Each channel is
// saturated at maxVolumeOpticalDepth, giving every (possibly pathological)
// ray a finite, bounded result.
func (v *Volume) OpticalDepth(ray Ray, tmin, tmax float64) Color {
	m := v.mediumConfig()
	if m == nil {
		return Black
	}
	sigmaT := m.extinction()
	tau := Black
	saturated := false
	v.marchSegments(ray, tmin, tmax, func(seg segment) {
		if saturated {
			return
		}
		density := (seg.d0 + seg.d1) / 2
		dt := seg.t1 - seg.t0
		if !isFinite(dt) || dt < 0 {
			return
		}
		inc := sigmaT.MulScalar(density * dt)
		tau = tau.Add(inc)
		tau = Color{
			R: math.Min(tau.R, maxVolumeOpticalDepth),
			G: math.Min(tau.G, maxVolumeOpticalDepth),
			B: math.Min(tau.B, maxVolumeOpticalDepth),
		}
		if tau.R >= maxVolumeOpticalDepth && tau.G >= maxVolumeOpticalDepth && tau.B >= maxVolumeOpticalDepth {
			saturated = true
		}
	})
	return tau
}

// Transmittance returns the cumulative beam transmittance over
// [tmin, tmax]: exp(-integral sigma_t density dt).
func (v *Volume) Transmittance(ray Ray, tmin, tmax float64) Color {
	tau := v.OpticalDepth(ray, tmin, tmax)
	return Color{math.Exp(-tau.R), math.Exp(-tau.G), math.Exp(-tau.B)}
}
