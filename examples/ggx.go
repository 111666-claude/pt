package main

import (
	"math/rand"

	. "github.com/fogleman/pt/pt"
)

// ggx.go renders a row of metallic and dielectric spheres whose microfacet
// roughness ramps from a perfect mirror (0) to a rough surface (0.8),
// demonstrating the GGX normal distribution and Smith masking/shadowing.
func main() {
	rand.Seed(1)
	scene := Scene{Color: HexColor(0x101014)}
	roughnesses := []float64{0, 0.1, 0.25, 0.45, 0.8}

	// metallic row: the color is the conductor's base reflectance
	for i, r := range roughnesses {
		x := float64(i-2) * 1.6
		scene.Add(NewSphere(V(x, 1, 0), 0.9, MetalMaterial(HexColor(0xC8C8D0), r)))
	}
	// dielectric row: Fresnel reflectance is derived from the index
	for i, r := range roughnesses {
		x := float64(i-2) * 1.6
		scene.Add(NewSphere(V(x, 1, -3.2), 0.9, DielectricMaterial(HexColor(0x4060A0), 1.5, r)))
	}

	scene.Add(NewCube(V(-20, -0.1, -20), V(20, 0, 20), DiffuseMaterial(HexColor(0x202024))))
	scene.Add(NewSphere(V(-3, 4, 3), 0.35, LightMaterial(White, 45)))
	scene.Add(NewSphere(V(3, 3.5, -2), 0.3, LightMaterial(HexColor(0xCFE0FF), 30)))

	camera := LookAt(V(0, 2.2, 7.5), V(0, 0.9, -1.2), V(0, 1, 0), 45)
	sampler := NewSampler(16, 5)
	sampler.SpecularMode = SpecularModeAll
	sampler.LightMode = LightModeAll
	renderer := NewRenderer(&scene, &camera, sampler, 640, 360)
	renderer.NumCPU = 4
	if err := SavePNG("ggx.png", renderer.Render()); err != nil {
		panic(err)
	}
}
