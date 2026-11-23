// Heterogeneous participating-media demo: a two-peak density volume rendered
// with deterministic beam transmittance, next-event-estimated in-scattering
// and weighted delta tracking for multiple scattering.
//
// Run with: go run ./examples/hetero
package main

import (
	"math"

	"github.com/fogleman/pt/pt"
)

func main() {
	// Build a 64^3 grid with two narrow Gaussian density peaks along x.
	const n = 64
	data := make([]float64, n*n*n)
	for z := 0; z < n; z++ {
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				wx := -1 + 2*(float64(x)+0.5)/float64(n)
				d := math.Exp(-(wx+0.45)*(wx+0.45)/(2*0.05*0.05)) +
					0.6*math.Exp(-(wx-0.45)*(wx-0.45)/(2*0.05*0.05))
				data[x+y*n+z*n*n] = d
			}
		}
	}
	medium := &pt.Medium{
		Step:             0.01,
		Absorption:       pt.Color{R: 0.25, G: 0.1, B: 0.05},
		Scattering:       pt.Color{R: 0.7, G: 0.8, B: 0.9},
		Phase:            pt.HenyeyGreensteinPhase{G: 0.4},
		MaxScatterEvents: 4,
	}
	volume := pt.NewVolumeData(pt.Box{Min: pt.V(-1, -1, -1), Max: pt.V(1, 1, 1)},
		n, n, n, data, medium)

	scene := pt.Scene{Color: pt.HexColor(0x0b0b12)}
	scene.Add(volume)
	scene.Add(pt.NewSphere(pt.V(1.6, 1.0, 0.8), 0.12, pt.LightMaterial(pt.White, 40)))

	camera := pt.LookAt(pt.V(0, -2.2, -2.6), pt.V(0, 0, 0), pt.V(0, 0, -1), 45)
	sampler := pt.NewSampler(4, 4)
	sampler.LightMode = pt.LightModeAll
	renderer := pt.NewRenderer(&scene, &camera, sampler, 400, 400)
	renderer.Verbose = true
	if err := pt.SavePNG("hetero.png", renderer.Render()); err != nil {
		panic(err)
	}
}
