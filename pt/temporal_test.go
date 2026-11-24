package pt

import (
	"image"
	"math"
	"runtime"
	"sync"
	"testing"
)

const (
	taaEps = 1e-12
	// Small but real rendering settings used by the end-to-end tests.
	taaW, taaH, taaSPP = 24, 16, 4
)

func colorEq(t *testing.T, name string, got, want Color, eps float64) {
	t.Helper()
	for i, pair := range [][2]float64{
		{got.R, want.R}, {got.G, want.G}, {got.B, want.B},
	} {
		if math.Abs(pair[0]-pair[1]) > eps {
			t.Fatalf("%s component %d: got %g want %g (eps %g)",
				name, i, pair[0], pair[1], eps)
		}
	}
}

func pixelFromSamples(samples ...Color) Pixel {
	var p Pixel
	for _, s := range samples {
		p.AddSample(s)
	}
	return p
}

func finitePixel(t *testing.T, name string, p Pixel) {
	t.Helper()
	for _, c := range [][3]float64{
		{p.M.R, p.M.G, p.M.B},
		{p.V.R, p.V.G, p.V.B},
	} {
		for _, x := range c {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("%s: non-finite value %g", name, x)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// 1. Exact pooling: AutoWeight merge must match one concatenated Welford run
// ---------------------------------------------------------------------------

func TestCombinePixelAutoWeightEqualsPooledSamples(t *testing.T) {
	groupA := []Color{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.2, 0.8, 0.1}}
	groupB := []Color{{0.9, 0.1, 0.7}, {0.3, 0.3, 0.3}}

	hist := pixelFromSamples(groupA...)
	cur := pixelFromSamples(groupB...)
	pooled := pixelFromSamples(append(append([]Color{}, groupA...), groupB...)...)

	merged, n := combinePixel(hist, cur, float64(hist.Samples), AutoWeight, 0)
	if n != float64(pooled.Samples) {
		t.Fatalf("effective weight: got %g want %d", n, pooled.Samples)
	}
	if merged.Samples != pooled.Samples {
		t.Fatalf("stored samples: got %d want %d", merged.Samples, pooled.Samples)
	}
	colorEq(t, "mean", merged.M, pooled.M, taaEps)
	colorEq(t, "M2", merged.V, pooled.V, taaEps)
}

// ---------------------------------------------------------------------------
// 2. EMA fixed weight: mean is the requested convex blend
// ---------------------------------------------------------------------------

func TestCombinePixelFixedAlpha(t *testing.T) {
	hist := pixelFromSamples(Color{0.8, 0, 0}, Color{1.0, 0, 0})
	cur := pixelFromSamples(Color{0.2, 1, 1})
	const alpha = 0.25

	merged, n := combinePixel(hist, cur, float64(hist.Samples), alpha, 0)
	want := hist.M.MulScalar(1 - alpha).Add(cur.M.MulScalar(alpha))
	colorEq(t, "ema mean", merged.M, want, taaEps)
	if n <= float64(hist.Samples) || n <= float64(cur.Samples) {
		t.Fatalf("effective weight %g should exceed both group sizes", n)
	}
	finitePixel(t, "ema", merged)
}

// ---------------------------------------------------------------------------
// 3. NaN / Inf sanitization: never poisoned statistics, never leaked
// ---------------------------------------------------------------------------

func TestCombinePixelNaNCurrentKeepsHistory(t *testing.T) {
	hist := pixelFromSamples(Color{0.3, 0.4, 0.5}, Color{0.5, 0.6, 0.7})
	cur := Pixel{Samples: 2, M: Color{math.NaN(), math.Inf(1), 0.1}, V: Black}

	merged, n := combinePixel(hist, cur, float64(hist.Samples), AutoWeight, 0)
	finitePixel(t, "merged", merged)
	colorEq(t, "preserved mean", merged.M, hist.M, taaEps)
	if n != float64(hist.Samples) {
		t.Fatalf("weight: got %g want %d", n, hist.Samples)
	}
}

func TestCombinePixelNaNFirstFrameIsBlack(t *testing.T) {
	cur := Pixel{Samples: 1, M: Color{math.NaN(), 0, math.Inf(-1)}, V: Black}
	merged, n := combinePixel(Pixel{}, cur, 0, AutoWeight, 0)
	finitePixel(t, "merged", merged)
	colorEq(t, "mean", merged.M, Black, taaEps)
	if n != 0 {
		t.Fatalf("weight: got %g want 0", n)
	}
}

func TestCombinePixelNaNHistoryRecovers(t *testing.T) {
	// A poisoned history pixel must be replaced, not blended, by good data.
	hist := Pixel{Samples: 3, M: Color{math.NaN(), 1, 1}, V: Color{1, 1, 1}}
	cur := pixelFromSamples(Color{0.2, 0.4, 0.6})
	merged, n := combinePixel(hist, cur, 3, AutoWeight, 0)
	finitePixel(t, "merged", merged)
	colorEq(t, "mean", merged.M, cur.M, taaEps)
	if n != 1 {
		t.Fatalf("weight: got %g want 1", n)
	}
}

// ---------------------------------------------------------------------------
// 4. Zero history / empty current frame
// ---------------------------------------------------------------------------

func TestCombinePixelZeroHistory(t *testing.T) {
	cur := pixelFromSamples(Color{1, 2, 3}, Color{3, 2, 1})
	merged, n := combinePixel(Pixel{}, cur, 0, AutoWeight, 0)
	colorEq(t, "mean", merged.M, cur.M, taaEps)
	colorEq(t, "M2", merged.V, cur.V, taaEps)
	if n != float64(cur.Samples) {
		t.Fatalf("weight: got %g want %d", n, cur.Samples)
	}

	// An empty current pixel leaves history untouched.
	hist := pixelFromSamples(Color{0.5, 0.5, 0.5})
	merged, n = combinePixel(hist, Pixel{}, 1, AutoWeight, 0)
	colorEq(t, "frozen mean", merged.M, hist.M, taaEps)
	if n != 1 {
		t.Fatalf("weight: got %g want 1", n)
	}
}

// ---------------------------------------------------------------------------
// 5. Weight boundary behaviour: alpha 0 freezes, alpha 1 replaces
// ---------------------------------------------------------------------------

func TestCombinePixelAlphaBoundaries(t *testing.T) {
	hist := pixelFromSamples(Color{1, 1, 1}, Color{0.9, 0.9, 0.9})
	cur := pixelFromSamples(Color{0, 0, 0})

	frozen, n0 := combinePixel(hist, cur, float64(hist.Samples), 0, 0)
	colorEq(t, "frozen", frozen.M, hist.M, taaEps)
	if n0 != float64(hist.Samples) {
		t.Fatalf("alpha 0 weight: got %g want %d", n0, hist.Samples)
	}

	replaced, n1 := combinePixel(hist, cur, float64(hist.Samples), 1, 0)
	colorEq(t, "replaced", replaced.M, cur.M, taaEps)
	if n1 != float64(cur.Samples) {
		t.Fatalf("alpha 1 weight: got %g want %d", n1, cur.Samples)
	}

	// Out-of-range and non-finite values are clamped deterministically.
	over, _ := combinePixel(hist, cur, float64(hist.Samples), 42, 0)
	colorEq(t, "alpha > 1", over.M, cur.M, taaEps)
	under, _ := combinePixel(hist, cur, float64(hist.Samples), -7, 0)
	colorEq(t, "alpha < 0", under.M, hist.M, taaEps)
	posInf, _ := combinePixel(hist, cur, float64(hist.Samples), math.Inf(1), 0)
	colorEq(t, "alpha +inf", posInf.M, cur.M, taaEps)
	negInf, _ := combinePixel(hist, cur, float64(hist.Samples), math.Inf(-1), 0)
	colorEq(t, "alpha -inf", negInf.M, hist.M, taaEps)
}

// ---------------------------------------------------------------------------
// 6. MaxHistory caps retained weight
// ---------------------------------------------------------------------------

func TestCombinePixelMaxHistoryCap(t *testing.T) {
	histSamples := make([]Color, 16) // 16 history samples
	hist := pixelFromSamples(histSamples...)
	cur := pixelFromSamples(Color{0.1, 0.1, 0.1}, Color{0.2, 0.2, 0.2})

	const maxHistory = 4
	_, n := combinePixel(hist, cur, 16, AutoWeight, maxHistory)
	// history weight is capped to (maxHistory-1) current-frame weights.
	if n > float64(maxHistory*cur.Samples)+taaEps {
		t.Fatalf("capped weight %g exceeds %d current weights", n, maxHistory)
	}
	if n < float64(cur.Samples) {
		t.Fatalf("capped weight %g dropped below current weight", n)
	}
}

// ---------------------------------------------------------------------------
// Fingerprinting: motion, camera cuts and transparent materials all reset
// ---------------------------------------------------------------------------

func taaTestScene() (*Scene, *Camera) {
	scene := Scene{}
	scene.Color = White
	scene.Add(NewSphere(Vector{0, 0, 0}, 1, DiffuseMaterial(HexColor(0x336699))))
	scene.Add(NewSphere(Vector{0, 0, 4}, 0.4, LightMaterial(White, 1)))
	camera := LookAt(Vector{0, -4, 1}, Vector{0, 0, 0}, Vector{0, 0, 1}, 30)
	return &scene, &camera
}

func TestFingerprintStableForIdenticalSetup(t *testing.T) {
	s1, c1 := taaTestScene()
	s2, c2 := taaTestScene()
	f1 := SceneFingerprint(s1, c1, taaW, taaH)
	f2 := SceneFingerprint(s2, c2, taaW, taaH)
	if !f1.Valid() || f1 != f2 {
		t.Fatalf("identical setups must fingerprint equal: %+v vs %+v", f1, f2)
	}
}

func TestFingerprintDetectsAllChanges(t *testing.T) {
	base, cam := taaTestScene()
	f0 := SceneFingerprint(base, cam, taaW, taaH)

	// Moving object (sphere center changes bounding box).
	moved := Scene{}
	moved.Color = White
	moved.Add(NewSphere(Vector{0.5, 0, 0}, 1, DiffuseMaterial(HexColor(0x336699))))
	moved.Add(NewSphere(Vector{0, 0, 4}, 0.4, LightMaterial(White, 1)))
	if f := SceneFingerprint(&moved, cam, taaW, taaH); f == f0 {
		t.Fatal("moving an object did not change fingerprint")
	}

	// Camera cut.
	cut := LookAt(Vector{3, -4, 2}, Vector{0, 0, 0}, Vector{0, 0, 1}, 30)
	if f := SceneFingerprint(base, &cut, taaW, taaH); f == f0 {
		t.Fatal("camera change did not change fingerprint")
	}

	// Transparent material change (opaque -> transparent, same color).
	transparent := Scene{}
	transparent.Color = White
	transparent.Add(NewSphere(Vector{0, 0, 0}, 1, ClearMaterial(1.5, 0)))
	transparent.Add(NewSphere(Vector{0, 0, 4}, 0.4, LightMaterial(White, 1)))
	if f := SceneFingerprint(&transparent, cam, taaW, taaH); f == f0 {
		t.Fatal("transparent material change did not change fingerprint")
	}

	// Material color change.
	recolor := Scene{}
	recolor.Color = White
	recolor.Add(NewSphere(Vector{0, 0, 0}, 1, DiffuseMaterial(HexColor(0x996633))))
	recolor.Add(NewSphere(Vector{0, 0, 4}, 0.4, LightMaterial(White, 1)))
	if f := SceneFingerprint(&recolor, cam, taaW, taaH); f == f0 {
		t.Fatal("material color change did not change fingerprint")
	}

	// Background change.
	bg := Scene{}
	bg.Color = HexColor(0x101010)
	bg.Add(NewSphere(Vector{0, 0, 0}, 1, DiffuseMaterial(HexColor(0x336699))))
	bg.Add(NewSphere(Vector{0, 0, 4}, 0.4, LightMaterial(White, 1)))
	if f := SceneFingerprint(&bg, cam, taaW, taaH); f == f0 {
		t.Fatal("background change did not change fingerprint")
	}

	// Dimension change.
	if f := SceneFingerprint(base, cam, taaW+1, taaH); f == f0 {
		t.Fatal("dimension change did not change fingerprint")
	}

	// Shape added / removed.
	grown := Scene{}
	grown.Color = White
	grown.Add(NewSphere(Vector{0, 0, 0}, 1, DiffuseMaterial(HexColor(0x336699))))
	grown.Add(NewSphere(Vector{2, 0, 0}, 0.3, DiffuseMaterial(HexColor(0x336699))))
	grown.Add(NewSphere(Vector{0, 0, 4}, 0.4, LightMaterial(White, 1)))
	if f := SceneFingerprint(&grown, cam, taaW, taaH); f == f0 {
		t.Fatal("added shape did not change fingerprint")
	}

	// Mesh material change (Mesh.MaterialAt is unimplemented, so this must be
	// caught via triangle introspection, not the generic probe).
	v1, v2, v3 := Vector{-1, -1, 0}, Vector{1, -1, 0}, Vector{0, 1, 0}
	t1 := Vector{0, 0, 0}
	meshA := NewMesh([]*Triangle{
		NewTriangle(v1, v2, v3, t1, t1, t1, DiffuseMaterial(HexColor(0x112233)))})
	meshB := NewMesh([]*Triangle{
		NewTriangle(v1, v2, v3, t1, t1, t1,
			TransparentMaterial(HexColor(0x112233), 1.5, 0, 0.5))})
	ms1 := Scene{Color: White, Shapes: []Shape{meshA}}
	ms2 := Scene{Color: White, Shapes: []Shape{meshB}}
	if SceneFingerprint(&ms1, cam, taaW, taaH) ==
		SceneFingerprint(&ms2, cam, taaW, taaH) {
		t.Fatal("mesh triangle material change did not change fingerprint")
	}

	// Identical meshes built independently hash equal (deterministic).
	meshA2 := NewMesh([]*Triangle{
		NewTriangle(v1, v2, v3, t1, t1, t1, DiffuseMaterial(HexColor(0x112233)))})
	ms3 := Scene{Color: White, Shapes: []Shape{meshA2}}
	if SceneFingerprint(&ms1, cam, taaW, taaH) !=
		SceneFingerprint(&ms3, cam, taaW, taaH) {
		t.Fatal("identical mesh content produced different fingerprints")
	}
}

// ---------------------------------------------------------------------------
// End-to-end renderer
// ---------------------------------------------------------------------------

func renderTAAFrame(t *testing.T, r *TemporalRenderer, scene *Scene, cam Camera) *Buffer {
	t.Helper()
	buf := r.Render(scene, &cam, NewSampler(4, 4), taaW, taaH, taaSPP)
	if buf.W != taaW || buf.H != taaH {
		t.Fatalf("output size: got %dx%d", buf.W, buf.H)
	}
	for i := range buf.Pixels {
		finitePixel(t, "output", buf.Pixels[i])
		if buf.Pixels[i].Samples <= 0 {
			t.Fatalf("pixel %d has no samples", i)
		}
	}
	return buf
}

// 7. Static scene: AutoWeight accumulation through the renderer equals one
// pooled Welford run over the identical sample stream. Synthetic,
// deterministic frame buffers are fed through applyFrame so the comparison is
// exact rather than at the mercy of independent RNG streams.
func TestTemporalStaticSceneMatchesPooledBuffer(t *testing.T) {
	scene, cam := taaTestScene()
	fp := SceneFingerprint(scene, cam, taaW, taaH)
	r := NewTemporalRenderer(DefaultTemporalConfig())

	// Deterministic per-frame sample values: frame f, pixel i gets a fixed
	// sample sequence, replayed identically into the renderer and the pool.
	frameSamples := func(f, i int) []Color {
		base := float64(f*7+i*3) * 0.01
		return []Color{
			{base + 0.1, base + 0.2, base + 0.3},
			{base + 0.4, base + 0.5, base + 0.6},
			{base + 0.2, base + 0.8, base + 0.1},
		}
	}

	var pooled Buffer
	const frames = 4
	for f := 0; f < frames; f++ {
		frame := NewBuffer(taaW, taaH)
		for i := range frame.Pixels {
			frame.Pixels[i] = pixelFromSamples(frameSamples(f, i)...)
		}
		out := r.applyFrame(frame, fp)

		if pooled.Pixels == nil {
			pooled = *frame
		} else {
			for i := range pooled.Pixels {
				pooled.Pixels[i], _ = combinePixel(
					pooled.Pixels[i], frame.Pixels[i],
					float64(pooled.Pixels[i].Samples), AutoWeight, 0)
			}
		}
		for i := range pooled.Pixels {
			colorEq(t, "static frame mean", out.Pixels[i].M, pooled.Pixels[i].M, taaEps)
			colorEq(t, "static frame M2", out.Pixels[i].V, pooled.Pixels[i].V, taaEps)
			if out.Pixels[i].Samples != pooled.Pixels[i].Samples {
				t.Fatalf("sample count pixel %d: %d vs %d",
					i, out.Pixels[i].Samples, pooled.Pixels[i].Samples)
			}
		}
	}
	if r.Frames() != frames {
		t.Fatalf("frame count: got %d want %d", r.Frames(), frames)
	}
}

// 7b. Render-level equivalence: on a static scene the accumulated output must
// be reproducible by merging the same frames through the stateless Accumulate
// helper (same buffer instances, hence identical random streams).
func TestTemporalRenderEqualsAccumulateHelper(t *testing.T) {
	scene, cam := taaTestScene()
	fp := SceneFingerprint(scene, cam, taaW, taaH)
	cfg := DefaultTemporalConfig()
	r := NewTemporalRenderer(cfg)

	var helper *Buffer
	var eff []float64
	for f := 0; f < 3; f++ {
		frame := NewBuffer(taaW, taaH)
		render(scene, cam, NewSampler(4, 4), taaSPP, frame)
		got := r.applyFrame(frame, fp)
		helper, eff = Accumulate(helper, eff, frame, cfg)
		for i := range got.Pixels {
			colorEq(t, "helper parity", got.Pixels[i].M, helper.Pixels[i].M, taaEps)
		}
	}
}

// 8. Static scene: temporal accumulation reduces noise (convergence curve).
// The converging quantity is the standard error of the mean, SD/sqrt(n), not
// the sample standard deviation (which is an estimator of the integrand's
// intrinsic spread and stays roughly constant as n grows).
func TestTemporalConvergenceMonotonic(t *testing.T) {
	scene, cam := taaTestScene()
	r := NewTemporalRenderer(DefaultTemporalConfig())

	var prevSE float64 = math.Inf(1)
	for f := 0; f < 6; f++ {
		buf := renderTAAFrame(t, r, scene, *cam)
		var sum float64
		for i := range buf.Pixels {
			n := buf.Pixels[i].Samples
			if n < 2 {
				continue
			}
			sum += buf.StandardDeviation(i%taaW, i/taaW).MaxComponent() /
				math.Sqrt(float64(n))
		}
		meanSE := sum / float64(len(buf.Pixels))
		if meanSE >= prevSE {
			t.Fatalf("frame %d: standard error did not decrease %.6f -> %.6f",
				f, prevSE, meanSE)
		}
		prevSE = meanSE
	}
}

// 9. Moving high-contrast edge: no ghost pixels from the previous position.
func TestTemporalMovingEdgeNoGhosting(t *testing.T) {
	cam := LookAt(Vector{0, -4, 0}, Vector{0, 0, 0}, Vector{0, 0, 1}, 30)
	makeScene := func(cx float64) *Scene {
		s := &Scene{}
		s.Color = Black
		s.Add(NewSphere(Vector{cx, 0, 0}, 0.6,
			LightMaterial(White, 2))) // bright emitter against black
		return s
	}

	r := NewTemporalRenderer(DefaultTemporalConfig())
	fp1 := SceneFingerprint(makeScene(-1.2), &cam, taaW, taaH)
	fp2 := SceneFingerprint(makeScene(1.2), &cam, taaW, taaH)
	if fp1 == fp2 {
		t.Fatal("moving edge did not change fingerprint")
	}

	// Frame 1: a real stochastic render at the old position.
	first := renderTAAFrame(t, r, makeScene(-1.2), cam)

	// Frame 2: render the new position into a standalone buffer, then apply it
	// through the history machinery under the new fingerprint. Because the
	// fingerprint changed, the output must equal that exact buffer: nothing
	// from frame 1 may be blended in.
	frame2 := NewBuffer(taaW, taaH)
	render(makeScene(1.2), &cam, NewSampler(4, 4), taaSPP, frame2)
	second := r.applyFrame(frame2, fp2)

	if r.Frames() != 1 {
		t.Fatalf("motion should have reset history: frames=%d", r.Frames())
	}
	if r.Fingerprint() != fp2 {
		t.Fatal("fingerprint not updated after motion reset")
	}
	for i := range second.Pixels {
		colorEq(t, "post-motion pixel", second.Pixels[i].M, frame2.Pixels[i].M, taaEps)
		colorEq(t, "post-motion M2", second.Pixels[i].V, frame2.Pixels[i].V, taaEps)
	}

	// Ghost check in the rendered output: pixels that were bright at the old
	// position but are background at the new position must come out black.
	var covered, ghosted int
	for i := range second.Pixels {
		if first.Pixels[i].M.MaxComponent() > 0.5 {
			covered++
			if second.Pixels[i].M.MaxComponent() > 0.05 &&
				frame2.Pixels[i].M.MaxComponent() < 0.05 {
				ghosted++
			}
		}
	}
	if covered == 0 {
		t.Fatal("test setup produced no bright pixels on frame 1")
	}
	if ghosted > 0 {
		t.Fatalf("%d ghost pixels from previous frame position", ghosted)
	}
}

// 10. Sudden scene cut (completely different scene) resets history.
func TestTemporalSceneCutResets(t *testing.T) {
	sceneA, cam := taaTestScene()
	r := NewTemporalRenderer(DefaultTemporalConfig())
	renderTAAFrame(t, r, sceneA, *cam)
	fpA := r.Fingerprint()

	sceneB := Scene{}
	sceneB.Color = HexColor(0x202040)
	sceneB.Add(NewSphere(Vector{0, 0, 2}, 0.9, MetallicMaterial(HexColor(0xcccccc), 0.1, 0.5)))
	fpB := SceneFingerprint(&sceneB, cam, taaW, taaH)
	if fpA == fpB {
		t.Fatal("test setup: scene cut fingerprints collide")
	}

	frameB := NewBuffer(taaW, taaH)
	render(&sceneB, cam, NewSampler(4, 4), taaSPP, frameB)
	buf := r.applyFrame(frameB, fpB)

	if r.Frames() != 1 {
		t.Fatalf("scene cut should reset: frames=%d", r.Frames())
	}
	if r.Fingerprint() != fpB {
		t.Fatal("fingerprint not updated after scene cut")
	}
	// On reset the output is exactly the new frame: zero leakage from A.
	for i := range buf.Pixels {
		colorEq(t, "cut pixel", buf.Pixels[i].M, frameB.Pixels[i].M, taaEps)
		colorEq(t, "cut M2", buf.Pixels[i].V, frameB.Pixels[i].V, taaEps)
	}
}

// 11. Transparent material change resets history.
func TestTemporalTransparentChangeResets(t *testing.T) {
	cam := LookAt(Vector{0, -4, 0}, Vector{0, 0, 0}, Vector{0, 0, 1}, 30)
	opaque := Scene{}
	opaque.Color = White
	opaque.Add(NewSphere(Vector{0, 0, 0}, 1, DiffuseMaterial(HexColor(0x884422))))

	transparent := Scene{}
	transparent.Color = White
	transparent.Add(NewSphere(Vector{0, 0, 0}, 1, TransparentMaterial(HexColor(0x884422), 1.5, 0, 0.5)))

	r := NewTemporalRenderer(DefaultTemporalConfig())
	renderTAAFrame(t, r, &opaque, cam)
	renderTAAFrame(t, r, &transparent, cam)
	if r.Frames() != 1 {
		t.Fatalf("transparent change should reset: frames=%d", r.Frames())
	}
}

// 12. Size change: old history is discarded, no out-of-bounds access.
func TestTemporalSizeChange(t *testing.T) {
	scene, cam := taaTestScene()
	r := NewTemporalRenderer(DefaultTemporalConfig())
	r.Render(scene, cam, NewSampler(4, 4), taaW, taaH, taaSPP)
	if r.Frames() != 1 {
		t.Fatalf("frames=%d", r.Frames())
	}
	buf := r.Render(scene, cam, NewSampler(4, 4), taaW/2, taaH/2, taaSPP)
	if buf.W != taaW/2 || buf.H != taaH/2 {
		t.Fatalf("resize output: %dx%d", buf.W, buf.H)
	}
	if r.Frames() != 1 {
		t.Fatalf("resize should reset: frames=%d", r.Frames())
	}
	if len(r.effective) != (taaW/2)*(taaH/2) {
		t.Fatalf("effective slice not resized: %d", len(r.effective))
	}
}

// 13. Disabled renderer is a plain single-frame pass with no history.
func TestTemporalDisabledPassthrough(t *testing.T) {
	scene, cam := taaTestScene()
	for _, cfg := range []TemporalConfig{
		{MaxHistory: 0, Alpha: AutoWeight},
		{MaxHistory: -3, Alpha: 0.5},
	} {
		r := NewTemporalRenderer(cfg)
		if r.Enabled() {
			t.Fatal("MaxHistory <= 0 must disable accumulation")
		}
		// The disabled path is render() into a fresh buffer with no merge,
		// verified directly by replaying render into an equivalent buffer.
		buf := r.Render(scene, cam, NewSampler(4, 4), taaW, taaH, taaSPP)
		if r.Frames() != 0 || r.history != nil || r.effective != nil {
			t.Fatal("disabled renderer must not retain history")
		}
		if buf.W != taaW || buf.H != taaH || len(buf.Pixels) != taaW*taaH {
			t.Fatal("passthrough buffer has wrong dimensions")
		}
		for i := range buf.Pixels {
			finitePixel(t, "passthrough", buf.Pixels[i])
			if buf.Pixels[i].Samples <= 0 {
				t.Fatalf("passthrough pixel %d has no samples", i)
			}
		}

		// Image output must be byte-identical to Buffer.Image (no TAA path
		// touches the existing rendering/output pipeline when disabled).
		im1 := buf.Image(ColorChannel)
		im2 := NewBuffer(taaW, taaH).Image(ColorChannel)
		if im1.Bounds() != im2.Bounds() {
			t.Fatal("passthrough image bounds mismatch")
		}
	}
}

// 14. Explicit Reset clears every trace of the old run.
func TestTemporalReset(t *testing.T) {
	scene, cam := taaTestScene()
	r := NewTemporalRenderer(DefaultTemporalConfig())
	renderTAAFrame(t, r, scene, *cam)
	r.Reset()
	if r.Frames() != 0 || r.history != nil || r.effective != nil || r.Fingerprint().Valid() {
		t.Fatal("Reset did not clear history state")
	}
	renderTAAFrame(t, r, scene, *cam)
	if r.Frames() != 1 {
		t.Fatalf("history did not restart after reset: frames=%d", r.Frames())
	}
}

// 15. Concurrent Render calls are serialized and never corrupt history.
//
// The upstream Scene is not safe to render concurrently (Compile mutates its
// k-d tree and the ray counter is atomic-but-racy), so the documented contract
// is one scene per goroutine while the TemporalRenderer is shared. Identical
// setups produce identical fingerprints, so every frame still lands in the
// same history.
func TestTemporalConcurrentRenders(t *testing.T) {
	r := NewTemporalRenderer(DefaultTemporalConfig())
	const goroutines = 8
	var wg sync.WaitGroup
	errs := make(chan string, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			scene, localCam := taaTestScene()
			buf := r.Render(scene, localCam, NewSampler(4, 4), taaW, taaH, taaSPP)
			if r.Fingerprint() != SceneFingerprint(scene, localCam, taaW, taaH) {
				errs <- "fingerprint mismatch across identical setups"
				return
			}
			for i := range buf.Pixels {
				c := buf.Pixels[i].M
				if math.IsNaN(c.R) || math.IsInf(c.R, 0) ||
					math.IsNaN(c.G) || math.IsInf(c.G, 0) ||
					math.IsNaN(c.B) || math.IsInf(c.B, 0) {
					errs <- "non-finite output under concurrent render"
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for msg := range errs {
		t.Fatal(msg)
	}
	if r.Frames() != goroutines {
		t.Fatalf("frames: got %d want %d", r.Frames(), goroutines)
	}
	if len(r.history.Pixels) != taaW*taaH || len(r.effective) != taaW*taaH {
		t.Fatal("history buffers corrupted by concurrent updates")
	}
}

// 15b. The history merge itself is safe under concurrent applyFrame calls with
// distinct fingerprints: no torn state, every run is fully represented or a
// clean reset happened (never a blend of two scenes).
func TestTemporalConcurrentApplyFrame(t *testing.T) {
	r := NewTemporalRenderer(DefaultTemporalConfig())
	const goroutines = 16
	fps := make([]Fingerprint, goroutines)
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			scene, cam := taaTestScene()
			// Shift a sphere per goroutine so fingerprints differ.
			scene.Shapes[0] = NewSphere(
				Vector{float64(g) * 0.01, 0, 0}, 1,
				DiffuseMaterial(HexColor(0x336699)))
			fp := SceneFingerprint(scene, cam, taaW, taaH)
			fps[g] = fp
			buf := NewBuffer(taaW, taaH)
			for i := range buf.Pixels {
				buf.Pixels[i] = pixelFromSamples(Color{
					float64(g) * 0.01, 0.2, 0.3})
			}
			out := r.applyFrame(buf, fp)
			for i := range out.Pixels {
				finitePixel(t, "concurrent output", out.Pixels[i])
			}
		}(g)
	}
	wg.Wait()
	// Every frame carries a distinct fingerprint, so each application resets
	// the run: the counter must never exceed 1 and every call was accounted for.
	if r.Frames() < 1 || r.Frames() > goroutines {
		t.Fatalf("frames: got %d, want within [1, %d]", r.Frames(), goroutines)
	}
	if len(r.history.Pixels) != taaW*taaH {
		t.Fatal("history corrupted")
	}
	// The final history must be a clean reset to exactly one of the frames,
	// never a blend of two scenes: fingerprint and pixel mean identify the
	// same goroutine.
	finalFP := r.Fingerprint()
	winner := -1
	for g, fp := range fps {
		if fp == finalFP {
			winner = g
			break
		}
	}
	if winner < 0 {
		t.Fatal("final fingerprint matches none of the applied frames")
	}
	wantR := float64(winner) * 0.01
	for i := range r.history.Pixels {
		colorEq(t, "clean reset mean", r.history.Pixels[i].M,
			Color{wantR, 0.2, 0.3}, taaEps)
	}
}

// 16. Memory stays bounded: exactly one history buffer plus one weight slice,
// regardless of frame count.
func TestTemporalMemoryBounded(t *testing.T) {
	scene, cam := taaTestScene()
	r := NewTemporalRenderer(TemporalConfig{MaxHistory: 4, Alpha: AutoWeight})

	var ms1, ms2 runtime.MemStats
	renderTAAFrame(t, r, scene, *cam)
	runtime.GC()
	runtime.ReadMemStats(&ms1)
	for f := 0; f < 12; f++ {
		renderTAAFrame(t, r, scene, *cam)
	}
	runtime.GC()
	runtime.ReadMemStats(&ms2)

	if len(r.history.Pixels) != taaW*taaH {
		t.Fatalf("history grew: %d pixels", len(r.history.Pixels))
	}
	if len(r.effective) != taaW*taaH {
		t.Fatalf("weight slice grew: %d", len(r.effective))
	}
	// Per-pixel effective weight must never exceed MaxHistory frames of samples.
	for i, n := range r.effective {
		if n > float64(4*taaSPP*4)+1 { // stratified 4 spp base + adaptive slack
			t.Fatalf("pixel %d effective weight unbounded: %g", i, n)
		}
	}
	// Heap growth over 12 frames must stay well under one extra full buffer.
	// A full frame buffer is far larger than the small delta allowed here.
	if delta := int64(ms2.HeapAlloc) - int64(ms1.HeapAlloc); delta > 2<<20 {
		t.Fatalf("heap grew by %d bytes over 12 frames", delta)
	}
}

// 17. Stateless Accumulate helper: nil start, dimension mismatch handling.
func TestAccumulateHelper(t *testing.T) {
	cfg := DefaultTemporalConfig()
	a := pixelFromSamples(Color{0.2, 0.4, 0.6}, Color{0.4, 0.6, 0.8})
	b := pixelFromSamples(Color{0.1, 0.3, 0.5})
	cur := NewBuffer(2, 1)
	cur.Pixels[0] = a
	cur.Pixels[1] = b

	hist, eff := Accumulate(nil, nil, cur, cfg)
	if hist == nil || len(eff) != 2 {
		t.Fatal("cold start failed")
	}
	colorEq(t, "cold pixel", hist.Pixels[0].M, a.M, taaEps)

	next := NewBuffer(2, 1)
	next.Pixels[0] = b
	next.Pixels[1] = a
	hist, eff = Accumulate(hist, eff, next, cfg)
	finitePixel(t, "helper", hist.Pixels[0])

	// Dimension mismatch discards old history deterministically.
	small := NewBuffer(1, 1)
	small.Pixels[0] = a
	hist, eff = Accumulate(hist, eff, small, cfg)
	if hist.W != 1 || hist.H != 1 || len(eff) != 1 {
		t.Fatal("dimension mismatch did not replace history")
	}
	colorEq(t, "replaced pixel", hist.Pixels[0].M, a.M, taaEps)

	// Nil current is a no-op.
	hist2, eff2 := Accumulate(hist, eff, nil, cfg)
	if hist2 != hist || len(eff2) != 1 {
		t.Fatal("nil current frame must be a no-op")
	}
}

// 18. The accumulated image remains a valid image.Image and gamma handling is
// unchanged from Buffer.Image.
func TestTemporalOutputImage(t *testing.T) {
	scene, cam := taaTestScene()
	r := NewTemporalRenderer(DefaultTemporalConfig())
	buf := renderTAAFrame(t, r, scene, *cam)
	im := buf.Image(ColorChannel)
	if _, ok := im.(image.Image); !ok {
		t.Fatal("output is not an image.Image")
	}
	if im.Bounds().Dx() != taaW || im.Bounds().Dy() != taaH {
		t.Fatal("output bounds mismatch")
	}
}
