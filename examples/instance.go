package main

import (
	"fmt"
	"math"
	"math/rand"

	. "github.com/fogleman/pt/pt"
)

// robot builds one shared robot mesh (a body, a head and a nose). Every
// robot in the scene is an Instance of this single mesh: no vertex data or
// k-d trees are duplicated, only a transform and an optional material
// override are stored per robot.
func robot() *Mesh {
	body := NewCube(V(-0.4, 0, -0.3), V(0.4, 0.8, 0.3), DiffuseMaterial(HexColor(0x334455)))
	head := NewCube(V(-0.25, 0.8, -0.25), V(0.25, 1.2, 0.25), DiffuseMaterial(HexColor(0x8899AA)))
	nose := NewCube(V(-0.08, 0.9, 0.25), V(0.08, 1.05, 0.4), DiffuseMaterial(HexColor(0xCC5533)))
	mesh := body.Mesh()
	mesh.Add(head.Mesh())
	mesh.Add(nose.Mesh())
	return mesh
}

func main() {
	scene := Scene{}

	// one shared mesh, thousands of instances
	mesh := robot()
	rnd := rand.New(rand.NewSource(1))
	const n = 40 // n x n robots
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x := (float64(i) - float64(n-1)/2) * 1.6
			z := (float64(j) - float64(n-1)/2) * 1.6
			angle := rnd.Float64() * 2 * math.Pi
			matrix := Identity().Rotate(V(0, 1, 0), angle).Translate(V(x, 0, z))
			instance, err := NewInstance(fmt.Sprintf("robot-%d-%d", i, j), mesh, matrix)
			if err != nil {
				panic(err)
			}
			// per-instance material override; the shared mesh is unaffected
			if (i+j)%7 == 0 {
				instance.SetMaterial(GlossyMaterial(HexColor(0xAA3322), 1.4, 0.2))
			}
			scene.Add(instance)
		}
	}

	// floor and light
	scene.Add(NewPlane(V(0, 0, 0), V(0, 1, 0), DiffuseMaterial(White)))
	scene.Add(NewSphere(V(6, 12, 10), 1.5, LightMaterial(White, 6)))

	camera := LookAt(V(0, 14, 24), V(0, 0.5, 0), V(0, 1, 0), 40)
	sampler := NewSampler(4, 4)
	renderer := NewRenderer(&scene, &camera, sampler, 960, 540)
	renderer.AdaptiveSamples = 64
	renderer.IterativeRender("instance%03d.png", 1000)
}
