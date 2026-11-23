package pt

import (
	"math"
	"math/rand"
)

// DefaultEnvMapResolution bounds the number of texels per axis when the
// importance sampling distribution is built from an environment texture.
// Building at full 8K resolution would waste memory and cache on every
// render, so textures are downsampled to at most this many columns/rows.
// 512 keeps per-texel solid-angle discretization error small (well under
// a percent for typical HDRIs) while capping the 2D CDF at ~262k entries.
const DefaultEnvMapResolution = 512

// EnvDistribution is a reusable, solid-angle-aware 2D discrete distribution
// over an equirectangular (latitude/longitude) environment texture.
//
// Each texel's weight is its luminance multiplied by the solid angle it
// covers on the unit sphere, so sampling the distribution picks directions
// proportionally to the radiance they carry. It is built once per scene and
// shared read-only by every rendering goroutine; all of its methods are
// safe for concurrent use after construction.
//
// Coordinate convention: v is the latitude fraction in [0,1] defined exactly
// as in DefaultSampler.sampleEnvironment, v = (lat + pi/2)/pi, and u is the
// longitude fraction. ColorTexture.Sample owns the image-row (1-v) flip, so
// a direction's (u,v) is passed to it verbatim.
type EnvDistribution struct {
	dist          *Distribution2D
	width, height int
	textureAngle  float64
	// totalWeight is the unnormalized integral of luminance over the sphere
	totalWeight float64
}

// envLuminance returns Rec. 709 relative luminance, clamping non-finite and
// negative components so a single bad pixel can not poison the distribution.
func envLuminance(c Color) float64 {
	r, g, b := c.R, c.G, c.B
	if !isFinite(r) || r < 0 {
		r = 0
	}
	if !isFinite(g) || g < 0 {
		g = 0
	}
	if !isFinite(b) || b < 0 {
		b = 0
	}
	return 0.2126*r + 0.7152*g + 0.0722*b
}

// envGridDimensions picks a downsampled grid that never exceeds
// DefaultEnvMapResolution texels on either axis while keeping the source
// dimensions.
func envGridDimensions(srcW, srcH int) (int, int) {
	if srcW <= 0 || srcH <= 0 {
		return 0, 0
	}
	w, h := srcW, srcH
	if w > DefaultEnvMapResolution {
		w = DefaultEnvMapResolution
	}
	if h > DefaultEnvMapResolution {
		h = DefaultEnvMapResolution
	}
	return w, h
}

// NewEnvDistribution builds an importance sampling distribution for the
// given environment texture. The returned distribution may be invalid
// (Valid() == false) for nil, empty, fully black or degenerate textures;
// callers must then fall back to the ordinary environment path.
func NewEnvDistribution(texture Texture, textureAngle float64) *EnvDistribution {
	if texture == nil {
		return &EnvDistribution{}
	}
	srcW, srcH, srcData := envTexturePixels(texture)
	if srcW <= 0 || srcH <= 0 {
		return &EnvDistribution{textureAngle: textureAngle}
	}
	w, h := envGridDimensions(srcW, srcH)
	if w <= 0 || h <= 0 {
		return &EnvDistribution{textureAngle: textureAngle}
	}

	weights := make([]float64, w*h)
	dphi := 2 * math.Pi / float64(w)
	dv := 1.0 / float64(h)
	for j := 0; j < h; j++ {
		// Rows index latitude v directly (v grows from south to north),
		// matching the (u,v) handed to ColorTexture.Sample. The exact
		// solid angle of a latitude band follows from
		// domega = cos(lat) dlat dphi with dlat = pi*dv.
		v0 := float64(j) * dv
		v1 := v0 + dv
		lat0 := v0*math.Pi - math.Pi/2
		lat1 := v1*math.Pi - math.Pi/2
		bandSolidAngle := math.Sin(lat1) - math.Sin(lat0)
		if bandSolidAngle < 0 {
			bandSolidAngle = 0
		}
		for i := 0; i < w; i++ {
			c := envSampleTexel(texture, srcW, srcH, srcData, i, j, w, h)
			// every texel in a row owns an equal slice of the band's
			// solid angle; polar rows correctly carry vanishing weight.
			weights[j*w+i] = envLuminance(c) * bandSolidAngle * dphi
		}
	}

	dist := NewDistribution2D(weights, w, h)
	ed := &EnvDistribution{
		dist:         dist,
		width:        w,
		height:       h,
		textureAngle: textureAngle,
	}
	if dist.Valid() {
		ed.totalWeight = dist.marginals.Integral()
	}
	return ed
}

// envTexturePixels returns the raw pixel grid when the texture is a
// ColorTexture (the common case for environment maps), enabling fast
// downsampling without repeated bilinear interpolation.
func envTexturePixels(texture Texture) (int, int, []Color) {
	if ct, ok := texture.(*ColorTexture); ok {
		return ct.Width, ct.Height, ct.Data
	}
	return 0, 0, nil
}

// envSampleTexel returns the source color for grid cell (i,j). The grid row
// j is a latitude v fraction, and ColorTexture.Sample performs the image-row
// flip, so the texture is queried at (u,v) directly.
func envSampleTexel(texture Texture, srcW, srcH int, srcData []Color, i, j, w, h int) Color {
	if srcData != nil {
		// nearest-neighbor read of the underlying linear-light texels
		fx := (float64(i) + 0.5) / float64(w)
		fy := (float64(j) + 0.5) / float64(h) // latitude fraction
		// ColorTexture stores image rows top-down; image row = (1-v)*srcH
		imgY := int((1 - fy) * float64(srcH))
		x := int(fx * float64(srcW))
		if x >= srcW {
			x = srcW - 1
		}
		if imgY >= srcH {
			imgY = srcH - 1
		}
		if imgY < 0 {
			imgY = 0
		}
		return srcData[imgY*srcW+x]
	}
	u := (float64(i) + 0.5) / float64(w)
	v := (float64(j) + 0.5) / float64(h)
	return texture.Sample(u, v)
}

// Valid reports whether environment importance sampling is available.
func (e *EnvDistribution) Valid() bool {
	return e != nil && e.dist != nil && e.dist.Valid() && e.width > 0 && e.height > 0 &&
		isFinite(e.totalWeight) && e.totalWeight > 0
}

// envDirectionToUV converts a world-space direction to the equirectangular
// texture coordinates used by the scene, including the texture rotation.
// The v convention matches DefaultSampler.sampleEnvironment exactly.
func envDirectionToUV(d Vector, textureAngle float64) (u, v float64) {
	phi := math.Atan2(d.Z, d.X) + textureAngle
	horiz := math.Hypot(d.X, d.Z)
	lat := math.Atan2(d.Y, horiz) // latitude in [-pi/2, pi/2]
	u = (phi + math.Pi) / (2 * math.Pi)
	v = (lat + math.Pi/2) / math.Pi
	return u, v
}

// envUVToDirection is the inverse of envDirectionToUV.
func envUVToDirection(u, v, textureAngle float64) Vector {
	phi := u*2*math.Pi - math.Pi - textureAngle
	lat := v*math.Pi - math.Pi/2
	cosLat := math.Cos(lat)
	return Vector{
		cosLat * math.Cos(phi),
		math.Sin(lat),
		cosLat * math.Sin(phi),
	}
}

// Sample picks an importance-sampled environment direction. It returns the
// world-space direction, its probability density with respect to solid
// angle (1/steradian), and the two parameter coordinates. On an invalid
// distribution it returns a zero pdf so callers can detect the fallback case.
func (e *EnvDistribution) Sample(rnd *rand.Rand) (dir Vector, pdf float64, u, v float64) {
	if !e.Valid() {
		return Vector{}, 0, 0, 0
	}
	u, v, pdfUV := e.dist.Sample(rnd.Float64(), rnd.Float64())
	dir = envUVToDirection(u, v, e.textureAngle)
	pdf = e.pdfOmega(v, pdfUV)
	return dir, pdf, u, v
}

// texelSolidAngle returns the exact solid angle covered by one texel in
// latitude row j, using the same band formula as the build weights so that
// pdf normalization is exact rather than a midpoint approximation.
func (e *EnvDistribution) texelSolidAngle(row int) float64 {
	dv := 1.0 / float64(e.height)
	v0 := float64(row) * dv
	v1 := v0 + dv
	lat0 := v0*math.Pi - math.Pi/2
	lat1 := v1*math.Pi - math.Pi/2
	band := math.Sin(lat1) - math.Sin(lat0)
	if band < 0 {
		band = 0
	}
	return band * (2 * math.Pi / float64(e.width))
}

// pdfOmega converts a joint parameter-domain density (per unit area of uv)
// into a solid-angle density (per steradian) at latitude coordinate v.
// Equating pdf_uv*du*dv = pdf_omega*domega with du*dv = 1/(w*h) gives
// pdf_omega = pdf_uv / (domega * w * h). The same per-texel solid angle
// used to build the weights is reused here, which makes the pdf integrate
// to 1 exactly.
func (e *EnvDistribution) pdfOmega(v, pdfUV float64) float64 {
	if !(pdfUV > 0) || !isFinite(pdfUV) {
		return 0
	}
	row := int(v * float64(e.height))
	if row >= e.height {
		row = e.height - 1
	}
	if row < 0 {
		row = 0
	}
	texelSolidAngle := e.texelSolidAngle(row)
	if texelSolidAngle <= 0 {
		// polar rows can collapse to zero solid angle; nothing finite can
		// be reported, treat as unsupported rather than divide by zero
		return 0
	}
	pdf := pdfUV / (texelSolidAngle * float64(e.width) * float64(e.height))
	if !isFinite(pdf) || pdf < 0 {
		return 0
	}
	return pdf
}

// PDF returns the solid-angle probability density of the given world
// direction, or 0 when the direction lands on a texel carrying no weight
// (or the distribution is invalid). It never divides by zero: polar rows
// whose solid angle collapses are reported as zero density.
func (e *EnvDistribution) PDF(dir Vector) float64 {
	if !e.Valid() {
		return 0
	}
	u, v := envDirectionToUV(dir, e.textureAngle)
	// Rotation can push u outside [0,1); wrap it exactly the way
	// ColorTexture.Sample does so the seam never produces a spurious zero.
	u = Fract(Fract(u) + 1)
	if v < 0 || v > 1 {
		return 0
	}
	return e.pdfOmega(v, e.dist.PDF(u, v))
}
