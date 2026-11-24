package pt

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
)

// ---------------------------------------------------------------------------
// Temporal anti-aliasing (TAA): accumulating samples across animation frames
//
// A TemporalRenderer renders one fresh frame Buffer per Render call and merges
// it into a history Buffer that is kept between frames. The merge reuses the
// per-pixel sample statistics (Welford running mean M and sum of squared
// deviations V) already maintained by Buffer, so the accumulated output stays
// a proper Monte Carlo estimate with meaningful variance instead of a
// frame-blurred image.
//
// A Fingerprint describes the camera/scene configuration that produced the
// history. Any change (moving object, camera cut, transparent material change,
// ...) invalidates the history, so pixels from an unrelated scene can never
// leak into the output.
// ---------------------------------------------------------------------------

// AutoWeight requests a sample-count-derived history weight: history and
// current samples are pooled exactly, which makes a static sequence converge
// to the same result as rendering all samples in a single frame.
var AutoWeight = math.NaN()

// TemporalConfig controls temporal accumulation.
//
// MaxHistory is the maximum number of rendered frames whose samples are
// retained per pixel (including the current frame). 0 or negative disables
// accumulation and Render behaves exactly like the single-frame Render.
//
// Alpha is the blend weight of the CURRENT frame, in [0, 1]. With AutoWeight
// the weight is derived from the effective sample counts so that every sample
// counts equally (exact pooling, static-scene output equals a single frame
// with the same total sample budget). A fixed value gives an exponential
// moving average; values outside [0, 1] are clamped, and 1 forces a reset on
// every frame (no history).
type TemporalConfig struct {
	MaxHistory int
	Alpha      float64
}

// DefaultTemporalConfig retains up to 8 frames with sample-count-derived
// weighting.
func DefaultTemporalConfig() TemporalConfig {
	return TemporalConfig{MaxHistory: 8, Alpha: AutoWeight}
}

// Fingerprint is a compact, value-comparable description of everything that
// can affect a rendered frame. A changed fingerprint means the stored history
// belongs to a different camera/scene configuration and must be discarded.
type Fingerprint struct {
	Width  int
	Height int
	Hash   uint64
}

// Valid reports whether a fingerprint belongs to a live history buffer.
func (f Fingerprint) Valid() bool {
	return f.Hash != 0
}

const fingerprintVersion = 0x54414131 // "TAA1"

// SceneFingerprint computes the fingerprint for the given frame setup. The
// hash covers camera parameters, scene background/texture, and every shape's
// transform (bounding box), material (including the Transparent flag) and
// texture identity. It is deterministic across runs and goroutine order.
func SceneFingerprint(scene *Scene, camera *Camera, w, h int) Fingerprint {
	var sb strings.Builder
	fmt.Fprintf(&sb, "v%d|%d,%d|", fingerprintVersion, w, h)

	// Camera: all fields that influence CastRay.
	fmt.Fprintf(&sb, "c(%g,%g,%g|%g,%g,%g|%g,%g,%g|%g,%g,%g|%g|%g,%g)",
		camera.p.X, camera.p.Y, camera.p.Z,
		camera.u.X, camera.u.Y, camera.u.Z,
		camera.v.X, camera.v.Y, camera.v.Z,
		camera.w.X, camera.w.Y, camera.w.Z,
		camera.m,
		camera.focalDistance, camera.apertureRadius)
	fmt.Fprintf(&sb, "|s(%g,%g,%g|%d|%d)",
		scene.Color.R, scene.Color.G, scene.Color.B,
		len(scene.Shapes), len(scene.Lights))
	if scene.Texture != nil {
		fmt.Fprintf(&sb, "|st(%s,%p)", typeName(scene.Texture), scene.Texture)
	}
	fmt.Fprintf(&sb, "|ta%g", scene.TextureAngle)

	for i, shape := range scene.Shapes {
		box := shape.BoundingBox()
		fmt.Fprintf(&sb, "|%d:%s[%g,%g,%g,%g,%g,%g]",
			i, typeName(shape),
			box.Min.X, box.Min.Y, box.Min.Z,
			box.Max.X, box.Max.Y, box.Max.Z)
		writeShapeMaterialFingerprint(&sb, shape, box)
	}

	return Fingerprint{Width: w, Height: h, Hash: hashString(sb.String())}
}

// writeShapeMaterialFingerprint covers the material-visible surface of a
// shape. Mesh.MaterialAt is unimplemented in this codebase, so meshes are
// introspected through their triangles; a transformed shape delegates to its
// inner shape.
func writeShapeMaterialFingerprint(sb *strings.Builder, shape Shape, box Box) {
	switch s := shape.(type) {
	case *Mesh:
		fmt.Fprintf(sb, "mesh(%d)", len(s.Triangles))
		// Mesh.MaterialAt is unimplemented, so every triangle's material is
		// summarized explicitly; material edits cannot hide inside a mesh.
		for j, tri := range s.Triangles {
			if tri == nil || tri.Material == nil {
				fmt.Fprintf(sb, "|t%d:nil", j)
				continue
			}
			writeMaterialFingerprint(sb, *tri.Material)
		}
	case *TransformedShape:
		writeShapeMaterialFingerprint(sb, s.Shape, s.Shape.BoundingBox())
	default:
		writeMaterialFingerprint(sb, shape.MaterialAt(box.Min))
	}
}

func writeMaterialFingerprint(sb *strings.Builder, m Material) {
	fmt.Fprintf(sb, "m(%g,%g,%g|e%g|i%g|g%g|t%g|r%g|tr%v|b%g)",
		m.Color.R, m.Color.G, m.Color.B,
		m.Emittance, m.Index, m.Gloss, m.Tint, m.Reflectivity,
		m.Transparent, m.BumpMultiplier)
	writeTextureFingerprint(sb, "tex", m.Texture)
	writeTextureFingerprint(sb, "nrm", m.NormalTexture)
	writeTextureFingerprint(sb, "bmp", m.BumpTexture)
	writeTextureFingerprint(sb, "glo", m.GlossTexture)
}

func writeTextureFingerprint(sb *strings.Builder, name string, t Texture) {
	if t == nil || reflect.ValueOf(t).IsNil() {
		sb.WriteString("|" + name + "0")
		return
	}
	// Identity (pointer) catches replacement; the type name catches a
	// different implementation behind the same interface.
	fmt.Fprintf(sb, "|%s(%s,%p)", name, typeName(t), t)
}

func typeName(v interface{}) string {
	return reflect.TypeOf(v).String()
}

// TemporalRenderer accumulates rendered frames over time. It is safe for
// concurrent Render calls: the history is updated under an internal mutex.
type TemporalRenderer struct {
	Config TemporalConfig

	mu        sync.Mutex
	history   *Buffer   // accumulated per-pixel statistics
	effective []float64 // effective sample weight per history pixel
	fp        Fingerprint
	frames    int // frames represented in the current history run
}

// NewTemporalRenderer creates a renderer with the given configuration.
func NewTemporalRenderer(config TemporalConfig) *TemporalRenderer {
	return &TemporalRenderer{Config: config}
}

// Enabled reports whether temporal accumulation is active.
func (r *TemporalRenderer) Enabled() bool {
	return r.Config.MaxHistory > 0
}

// Frames returns the number of frames represented by the current history
// (0 before the first render or after a reset).
func (r *TemporalRenderer) Frames() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.frames
}

// Fingerprint returns the fingerprint of the current history.
func (r *TemporalRenderer) Fingerprint() Fingerprint {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.fp
}

// Reset discards the accumulated history. The next Render starts a fresh run.
func (r *TemporalRenderer) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.resetLocked()
}

func (r *TemporalRenderer) resetLocked() {
	r.history = nil
	r.effective = nil
	r.fp = Fingerprint{}
	r.frames = 0
}

// Render renders one frame of the given setup and returns the accumulated
// output image. When accumulation is disabled the call is equivalent to the
// package-level Render and no history is retained.
//
// Concurrency: history updates are guarded by an internal mutex, so multiple
// goroutines may share a TemporalRenderer. The upstream Scene is itself not
// safe for concurrent rendering (Compile mutates its k-d tree), so each
// goroutine must pass its own Scene/Camera; identical setups hash to the same
// fingerprint and therefore accumulate into one shared history.
func (r *TemporalRenderer) Render(scene *Scene, camera *Camera, sampler Sampler,
	w, h, samplesPerPixel int) *Buffer {

	if !r.Enabled() {
		buf := NewBuffer(w, h)
		render(scene, camera, sampler, samplesPerPixel, buf)
		return buf
	}

	fp := SceneFingerprint(scene, camera, w, h)
	current := NewBuffer(w, h)
	render(scene, camera, sampler, samplesPerPixel, current)
	return r.applyFrame(current, fp)
}

// applyFrame merges an already-rendered frame buffer into the history under
// the given fingerprint. It is the single mutation point of the history and
// is also used directly by tests with synthetic, deterministic buffers.
func (r *TemporalRenderer) applyFrame(current *Buffer, fp Fingerprint) *Buffer {
	w, h := current.W, current.H
	r.mu.Lock()
	defer r.mu.Unlock()

	// A changed fingerprint, a missing history or a dimension mismatch means
	// the old pixels describe another scene: start over, never blend.
	if r.history == nil || r.fp != fp || r.history.W != w || r.history.H != h {
		r.resetLocked()
		r.history = NewBuffer(w, h)
		r.effective = make([]float64, w*h)
		r.fp = fp
	}

	alpha := r.resolveAlphaLocked(current)
	mergeBuffers(r.history, r.effective, current, alpha, r.Config.MaxHistory)
	r.frames++
	return r.history.Copy()
}

// resolveAlphaLocked turns the configured weight into a per-frame current
// weight in [0, 1]. AutoWeight yields the exact pooling weight so a static
// scene matches a single-frame render with the same total sample count.
func (r *TemporalRenderer) resolveAlphaLocked(current *Buffer) float64 {
	a := r.Config.Alpha
	if math.IsNaN(a) {
		var historySamples, currentSamples float64
		for i := range current.Pixels {
			historySamples += r.effective[i]
			currentSamples += float64(current.Pixels[i].Samples)
		}
		total := historySamples + currentSamples
		if total <= 0 {
			return 1
		}
		return currentSamples / total
	}
	if math.IsInf(a, 1) || a > 1 {
		return 1
	}
	if math.IsInf(a, -1) || a < 0 {
		return 0
	}
	return a
}

// ---------------------------------------------------------------------------
// Pixel-level merge (pure function, easy to test in isolation)
// ---------------------------------------------------------------------------

// finiteColor replaces non-finite components with 0 so NaN/Inf can never
// poison the running statistics or leak into an output image.
func finiteColor(c Color) (Color, bool) {
	ok := true
	clean := func(x float64) float64 {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			ok = false
			return 0
		}
		return x
	}
	return Color{clean(c.R), clean(c.G), clean(c.B)}, ok
}

// combinePixel merges a current pixel into an accumulated history pixel.
//
// effective is the (possibly fractional) sample weight represented by hist.
// alpha is the weight of the current frame in [0, 1]; maxHistory caps the
// total retained weight. Non-finite inputs are sanitized. The returned weight
// is the new effective weight of the merged pixel.
//
// The merge is the parallel (Chan) Welford combination: with AutoWeight on a
// static scene the result is algebraically identical to concatenating both
// sample streams in one Buffer.
func combinePixel(hist, cur Pixel, effective, alpha float64, maxHistory int) (Pixel, float64) {
	curColor, curOK := finiteColor(cur.M)
	histColor, histOK := finiteColor(hist.M)
	histV, _ := finiteColor(hist.V)
	curV, _ := finiteColor(cur.V)

	// Degenerate current frame: no samples or NaN-only input. Keep history
	// verbatim (or black on the first frame) so a bad frame cannot corrupt
	// an established estimate.
	if cur.Samples <= 0 || !curOK {
		hist.M = histColor
		hist.V = histV
		if hist.Samples <= 0 {
			return hist, 0
		}
		return hist, effective
	}

	nB := float64(cur.Samples)
	nA := effective
	if hist.Samples <= 0 || !histOK {
		nA = 0
	}

	// Fixed-weight EMA: express alpha as an equivalent history weight while
	// preserving the current frame's real sample count.
	if !math.IsNaN(alpha) {
		if alpha >= 1 || nA <= 0 {
			nA = 0
		} else if alpha <= 0 {
			nB = 0 // current frame contributes nothing
		} else {
			nA = nB * (1 - alpha) / alpha
		}
	}

	// No usable history: the current pixel stands alone.
	if nA <= 0 {
		return Pixel{Samples: cur.Samples, M: curColor, V: curV}, nB
	}
	// Alpha 0: history is frozen, current frame discarded.
	if nB <= 0 {
		hist.M = histColor
		hist.V = histV
		return hist, nA
	}

	// Bound the retained history weight so old frames cannot dominate and
	// memory/statistics stay bounded by MaxHistory frames.
	if maxHistory > 0 {
		if cap := float64(maxHistory-1) * nB; nA > cap {
			nA = cap
		}
	}

	n := nA + nB
	delta := curColor.Sub(histColor)
	mean := histColor.Add(delta.MulScalar(nB / n))
	// Parallel Welford M2 update, generalized to fractional group weights.
	m2 := histV.Add(curV).Add(
		delta.Mul(delta).MulScalar(nA * nB / n))

	// Stored Samples drives the existing variance estimator (V / (Samples-1));
	// round the fractional weight so V/(n-1) remains a valid variance.
	samples := int(math.Round(n))
	if samples < 2 {
		samples = 2
	}
	return Pixel{Samples: samples, M: mean, V: m2}, n
}

// mergeBuffers applies combinePixel over every pixel of a frame.
func mergeBuffers(history *Buffer, effective []float64, current *Buffer, alpha float64, maxHistory int) {
	for i := range history.Pixels {
		history.Pixels[i], effective[i] = combinePixel(
			history.Pixels[i], current.Pixels[i], effective[i], alpha, maxHistory)
	}
}

// hashString returns the 64-bit FNV-1a hash of s, used to keep fingerprints
// small and value-comparable without carrying the full canonical string.
func hashString(s string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime64
	}
	return h
}

// Accumulate is a stateless convenience for tests and batch pipelines: it
// merges current into history and returns the merged buffer and its effective
// per-pixel weights. A nil history starts a fresh run. Dimension mismatch
// discards the history.
func Accumulate(history *Buffer, effective []float64, current *Buffer, config TemporalConfig) (*Buffer, []float64) {
	if current == nil {
		return history, effective
	}
	if history == nil || history.W != current.W || history.H != current.H {
		history = NewBuffer(current.W, current.H)
		effective = make([]float64, current.W*current.H)
	}
	alpha := config.Alpha
	if math.IsNaN(alpha) {
		var hs, cs float64
		for i := range current.Pixels {
			hs += effective[i]
			cs += float64(current.Pixels[i].Samples)
		}
		if hs+cs > 0 {
			alpha = cs / (hs + cs)
		} else {
			alpha = 1
		}
	}
	mergeBuffers(history, effective, current, alpha, config.MaxHistory)
	return history, effective
}
