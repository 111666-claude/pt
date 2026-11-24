# pt: a golang path tracer

[![Build Status](https://travis-ci.org/fogleman/pt.png?branch=master)](https://travis-ci.org/fogleman/pt) [![GoDoc](https://godoc.org/github.com/fogleman/pt/pt?status.svg)](https://godoc.org/github.com/fogleman/pt/pt)

This is a CPU-only, unidirectional [path tracing](http://en.wikipedia.org/wiki/Path_tracing) engine written in Go. It has lots of features and a simple API.

![Examples](http://i.imgur.com/BTaDLC3.png)

---

### Features

* Supports OBJ and STL
* Supports textures, bump maps and normal maps
* Supports raymarching of signed distance fields
* Supports volume rendering from image slices
* Supports various material properties
* Supports configurable depth of field
* Supports iterative rendering
* Supports temporal anti-aliasing with history accumulation and automatic
  invalidation on scene/camera/material changes
* Uses k-d trees to accelerate ray intersection tests
* Uses all CPU cores in parallel
* 100% pure Go with no dependencies besides the standard library

### Installation

    go get -u github.com/fogleman/pt/pt

### Examples

The are [lots of examples](https://github.com/fogleman/pt/tree/master/examples) to learn from! To try them, just run, e.g.

    cd go/src/github.com/fogleman/pt
    go run examples/gopher.go

### Optional Embree Acceleration

You can optionally utilize Intel's Embree ray tracing kernels to accelerate triangle mesh intersections. First, install embree on your system: http://embree.github.io/ Then get the `go-embree` wrapper and checkout the `embree` branch of `pt`.

    git checkout embree
    go get -u github.com/fogleman/go-embree

### Hello World

The following code demonstrates the basics of the API.

```go
package main

import . "github.com/fogleman/pt/pt"

func main() {
	// create a scene
	scene := Scene{}

	// create a material
	material := DiffuseMaterial(White)

	// add the floor (a plane)
	plane := NewPlane(V(0, 0, 0), V(0, 0, 1), material)
	scene.Add(plane)

	// add the ball (a sphere)
	sphere := NewSphere(V(0, 0, 1), 1, material)
	scene.Add(sphere)

	// add a spherical light source
	light := NewSphere(V(0, 0, 5), 1, LightMaterial(White, 8))
	scene.Add(light)

	// position the camera
	camera := LookAt(V(3, 3, 3), V(0, 0, 0.5), V(0, 0, 1), 50)

	// render the scene with progressive refinement
	sampler := NewSampler(4, 4)
	IterativeRender("out%03d.png", 1000, &scene, &camera, sampler, 960, 540, -1)
}
```

![Hello World](http://i.imgur.com/0TXY0dX.png)

### Temporal Anti-Aliasing

Animation frames rendered independently flicker, especially along moving
high-contrast edges. `TemporalRenderer` accumulates samples across frames by
reusing the `Buffer`'s per-pixel Welford running statistics, so the output
remains a proper Monte Carlo estimate whose standard error shrinks as more
frames are pooled — not a frame-blurred image.

```go
// Retain up to 8 frames; AutoWeight pools every sample exactly, so a static
// scene converges identically to one render with the same total sample budget.
renderer := NewTemporalRenderer(DefaultTemporalConfig())

for each animation frame {
    scene := buildScene(frameTime)   // moving objects, camera cuts, ...
    camera := buildCamera(frameTime)
    buf := renderer.Render(scene, camera, sampler, width, height, spp)
    SavePNG(fmt.Sprintf("frame%03d.png", i), buf.Image(ColorChannel))
}
```

**Fingerprinting and automatic reset.** Before each merge a `Fingerprint`
hashes everything that affects the image: camera parameters, background, and
every shape's bounding box (transform), full material (including the
`Transparent` flag, refractive index, ...) and texture identity. A moving
object, a camera cut, a transparent-material change or a dimension change all
change the fingerprint, and the history is then discarded instead of blended —
old frames can never leak into a new scene. On a static setup the fingerprint
is stable and the pooled result equals a single-frame render with the same
sample count.

**Configuration.** `TemporalConfig{MaxHistory: n, Alpha: a}`:

* `MaxHistory` bounds the retained sample weight per pixel; `0` or negative
  disables accumulation and `Render` is a plain single-frame pass.
* `Alpha` is the current frame's blend weight in `[0,1]` (out-of-range and
  non-finite values are clamped). `AutoWeight` (the default) derives the
  weight from sample counts for exact pooling.

**Deterministic edge cases.** NaN/Inf sample colors are sanitized to black and
never poison the running mean or variance; a degenerate frame freezes the
history rather than corrupting it; `Reset()` clears all state; history updates
are mutex-guarded, so a `TemporalRenderer` may be shared by goroutines (each
still needs its own `Scene`, since the scene's k-d tree is not concurrency
safe).

The validation scenarios — moving high-contrast edge (ghost energy), static
control (1/√n convergence curve), abrupt cut (zero leakage) and long-run memory
bound — can be reproduced with:

    go run ./cmd/taacheck -out artifacts

### Links

Here are some resources that I have found useful.

* [WebGL Path Tracing - Evan Wallace](http://madebyevan.com/webgl-path-tracing/)
* [Global Illumination in a Nutshell](http://www.thepolygoners.com/tutorials/GIIntro/GIIntro.htm)
* [Simple Path Tracing - Iñigo Quilez](http://www.iquilezles.org/www/articles/simplepathtracing/simplepathtracing.htm)
* [Realistic Raytracing - Zack Waters](http://web.cs.wpi.edu/~emmanuel/courses/cs563/write_ups/zackw/realistic_raytracing.html)
* [Reflections and Refractions in Ray Tracing - Bram de Greve](http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf)
* [Better Sampling - Rory Driscoll](http://www.rorydriscoll.com/2009/01/07/better-sampling/)
* [Ray Tracing for Global Illumination - Nelson Max at UC Davis](https://www.youtube.com/playlist?list=PLslgisHe5tBPckSYyKoU3jEA4bqiFmNBJ)
* [Physically Based Rendering - Matt Pharr, Greg Humphreys](http://www.amazon.com/Physically-Based-Rendering-Second-Edition/dp/0123750792)

### Samples

![Go](http://i.imgur.com/LMNUoaM.jpg)

![Dragon](https://www.michaelfogleman.com/static/gallery/out1000c.png)

![Cornell](https://www.michaelfogleman.com/static/gallery/853.png)

![Lucy](https://www.michaelfogleman.com/static/gallery/756b.png)

![SDF](https://www.michaelfogleman.com/static/gallery/470d.png)

![Spheres](https://www.michaelfogleman.com/static/gallery/dof.png)

![Suzanne](http://i.imgur.com/iw32US1.png)

![Molecule](https://www.michaelfogleman.com/static/gallery/600d.png)
