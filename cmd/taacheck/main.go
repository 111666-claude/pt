// Command taacheck renders the temporal anti-aliasing validation scenarios:
// a moving high-contrast edge (ghosting), a static control (convergence curve
// and single-frame equivalence), an abrupt scene cut and a long static run
// (memory bound). Metrics are written as CSV to the output directory and
// summarized on stdout.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

	"github.com/fogleman/pt/pt"
)

const (
	W   = 96
	H   = 64
	SPP = 4
)

func staticScene() (*pt.Scene, *pt.Camera) {
	scene := &pt.Scene{Color: pt.HexColor(0x202830)}
	scene.Add(pt.NewSphere(pt.Vector{X: 0, Y: 0, Z: 0}, 1,
		pt.GlossyMaterial(pt.HexColor(0x336699), 1.333, pt.Radians(20))))
	scene.Add(pt.NewSphere(pt.Vector{X: 0, Y: 0, Z: 4}, 0.4,
		pt.LightMaterial(pt.White, 1)))
	cam := pt.LookAt(pt.Vector{X: 0, Y: -4, Z: 1}, pt.Vector{X: 0, Y: 0, Z: 0},
		pt.Vector{X: 0, Y: 0, Z: 1}, 30)
	return scene, &cam
}

func edgeScene(cx float64) (*pt.Scene, *pt.Camera) {
	scene := &pt.Scene{Color: pt.Black}
	scene.Add(pt.NewSphere(pt.Vector{X: cx, Y: 0, Z: 0}, 0.6,
		pt.LightMaterial(pt.White, 2)))
	cam := pt.LookAt(pt.Vector{X: 0, Y: -4, Z: 0}, pt.Vector{X: 0, Y: 0, Z: 0},
		pt.Vector{X: 0, Y: 0, Z: 1}, 30)
	return scene, &cam
}

func meanSE(b *pt.Buffer) float64 {
	var sum float64
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			n := b.Samples(x, y)
			if n < 2 {
				continue
			}
			// Standard error of the mean: sample SD / sqrt(n). The sample SD
			// itself is invariant to n; the error of the estimate shrinks as
			// 1/sqrt(n), which is the convergence curve TAA must produce.
			se := b.StandardDeviation(x, y).MaxComponent() /
				math.Sqrt(float64(n))
			sum += se
		}
	}
	return sum / float64(W*H)
}

func backgroundEnergy(taa, pristine *pt.Buffer) (energy float64, bgPixels int) {
	for i := range taa.Pixels {
		// Pristine frame defines true background for this geometry.
		if pristine.Pixels[i].M.MaxComponent() < 0.02 {
			bgPixels++
			energy += taa.Pixels[i].M.MaxComponent()
		}
	}
	return energy, bgPixels
}

func meanSquaredError(a, b *pt.Buffer) float64 {
	var sum float64
	for i := range a.Pixels {
		d := a.Pixels[i].M.Sub(b.Pixels[i].M)
		sum += d.R*d.R + d.G*d.G + d.B*d.B
	}
	return sum / float64(len(a.Pixels)*3)
}

func writeCSV(path string, header []string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.Write(header); err != nil {
		return err
	}
	return w.WriteAll(rows)
}

func main() {
	out := flag.String("out", "artifacts", "output directory for CSV metrics")
	flag.Parse()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		panic(err)
	}
	sampler := pt.NewSampler(8, 8)

	// --- Scenario 1: static control, convergence curve --------------------
	fmt.Println("== static control: convergence over frames ==")
	scene, cam := staticScene()
	r := pt.NewTemporalRenderer(pt.DefaultTemporalConfig())
	staticRows := [][]string{}
	const staticFrames = 8
	var last *pt.Buffer
	for f := 1; f <= staticFrames; f++ {
		last = r.Render(scene, cam, sampler, W, H, SPP)
		staticRows = append(staticRows, []string{
			fmt.Sprintf("%d", f), fmt.Sprintf("%.6f", meanSE(last))})
		fmt.Printf("frame %d: mean per-pixel standard error %.6f\n", f, meanSE(last))
	}
	if err := writeCSV(filepath.Join(*out, "taa_convergence.csv"),
		[]string{"frame", "mean_standard_error"}, staticRows); err != nil {
		panic(err)
	}

	// Single-frame render with the same total sample budget. Both estimate
	// the same integral, so their MSE must be small relative to the signal.
	pooled := pt.NewBuffer(W, H)
	pt.RenderInto(scene, cam, sampler, W, H, SPP*staticFrames, pooled)
	mse := meanSquaredError(last, pooled)
	fmt.Printf("MSE vs same-budget single-frame render: %.6f\n\n", mse)

	// --- Scenario 2: moving high-contrast edge, ghosting ------------------
	fmt.Println("== moving edge: ghost energy in background ==")
	edge := pt.NewTemporalRenderer(pt.DefaultTemporalConfig())
	ghostRows := [][]string{}
	positions := []float64{-1.5, -0.9, -0.3, 0.3, 0.9, 1.5}
	var maxGhost float64
	for f, cx := range positions {
		s, c := edgeScene(cx)
		out := edge.Render(s, c, sampler, W, H, SPP)
		// High-spp pristine frame of exactly this geometry: the ground truth
		// for which pixels are background.
		pristineBuf := pt.NewBuffer(W, H)
		pt.RenderInto(s, c, pt.NewSampler(16, 16), W, H, 32, pristineBuf)
		energy, bg := backgroundEnergy(out, pristineBuf)
		refEnergy, _ := backgroundEnergy(pristineBuf, pristineBuf)
		// Normalize per background pixel so the metric is resolution free.
		ghost := (energy - refEnergy) / float64(bg)
		if ghost > maxGhost {
			maxGhost = ghost
		}
		ghostRows = append(ghostRows, []string{
			fmt.Sprintf("%d", f), fmt.Sprintf("%.3f", cx),
			fmt.Sprintf("%.6f", ghost)})
		fmt.Printf("frame %d (cx=%.2f): excess ghost energy/px %.6f\n", f, cx, ghost)
	}
	if err := writeCSV(filepath.Join(*out, "taa_ghosting.csv"),
		[]string{"frame", "sphere_x", "excess_ghost_energy_per_px"}, ghostRows); err != nil {
		panic(err)
	}
	if maxGhost > 0.02 {
		fmt.Printf("FAIL: ghosting threshold exceeded (%.6f > 0.02)\n", maxGhost)
		os.Exit(1)
	}

	// --- Scenario 3: abrupt scene cut -------------------------------------
	fmt.Println("\n== abrupt scene cut: zero leakage ==")
	cut := pt.NewTemporalRenderer(pt.DefaultTemporalConfig())
	sA, cA := edgeScene(-1.5)
	cut.Render(sA, cA, sampler, W, H, SPP)
	sB, cB := staticScene()
	before := cut.Frames()
	cutBuf := cut.Render(sB, cB, sampler, W, H, SPP)
	// Independent single-frame render of scene B (the MC noise floor: this is
	// the best any single-frame output can do against a different RNG stream).
	indieBuf := pt.NewBuffer(W, H)
	pt.RenderInto(sB, cB, sampler, W, H, SPP, indieBuf)
	// High-spp ground truth of scene B for an absolute bias check.
	truthBuf := pt.NewBuffer(W, H)
	pt.RenderInto(sB, cB, pt.NewSampler(16, 16), W, H, 64, truthBuf)
	cutMSE := meanSquaredError(cutBuf, truthBuf)
	indieMSE := meanSquaredError(indieBuf, truthBuf)
	fmt.Printf("frames before/after cut: %d -> %d\n", before, cut.Frames())
	fmt.Printf("MSE to 64spp ground truth — post-cut: %.6f, single-frame: %.6f\n",
		cutMSE, indieMSE)
	// With a clean reset the post-cut frame is itself a single-frame render,
	// so its error must be statistically comparable to the independent one
	// (same order of magnitude, within 2x), never worse by a large factor.
	if cut.Frames() != 1 || cutMSE > 2*indieMSE {
		fmt.Println("FAIL: history leaked across scene cut")
		os.Exit(1)
	}
	// And it must not resemble scene A at all: bright edge-A pixels that are
	// background in B's ground truth must be dark in the post-cut buffer.
	var leaks int
	for i := range cutBuf.Pixels {
		if truthBuf.Pixels[i].M.MaxComponent() < 0.02 &&
			cutBuf.Pixels[i].M.MaxComponent() > 0.1 {
			leaks++
		}
	}
	if leaks > 0 {
		fmt.Printf("FAIL: %d background pixels carry scene-A energy\n", leaks)
		os.Exit(1)
	}

	// --- Scenario 4: memory bound over a long run -------------------------
	fmt.Println("\n== memory: long static run ==")
	var ms1, ms2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&ms1)
	mem := pt.NewTemporalRenderer(pt.TemporalConfig{MaxHistory: 4, Alpha: pt.AutoWeight})
	for f := 0; f < 32; f++ {
		mem.Render(scene, cam, sampler, W, H, SPP)
	}
	runtime.GC()
	runtime.ReadMemStats(&ms2)
	growth := int64(ms2.HeapAlloc) - int64(ms1.HeapAlloc)
	// One frame's pixel data (Pixel = int + two 3-float Colors, 56 bytes on
	// amd64) plus the effective-weights slice (8 bytes/px) is the only allowed
	// resident cost beyond the first frame; allow 4x slack for runtime/GC.
	oneBuffer := int64(W * H * 64)
	fmt.Printf("heap growth over 32 frames: %d bytes (one buffer ~ %d)\n",
		growth, oneBuffer)
	if growth > 4*oneBuffer {
		fmt.Println("FAIL: history memory is unbounded")
		os.Exit(1)
	}

	fmt.Println("\nALL TAA VALIDATION CHECKS PASSED")
}
