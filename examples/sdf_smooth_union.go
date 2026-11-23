package main

import . "github.com/fogleman/pt/pt"

// Demonstrates the smooth SDF union: two spheres merge with a rounded,
// continuous transition instead of the hard seam produced by NewUnionSDF.
// The children carry different materials via NewMaterialSDF, so the blended
// surface picks the material of the child governing each point.
func main() {
	scene := Scene{}

	light := LightMaterial(White, 180)
	scene.Add(NewSphere(V(-1, -1, 0.5).Normalize().MulScalar(4), 0.25, light))
	scene.Add(NewSphere(V(1, -1, 0.25).Normalize().MulScalar(4), 0.25, light))

	red := GlossyMaterial(HexColor(0xB8262A), 1.2, Radians(20))
	blue := GlossyMaterial(HexColor(0x2648B8), 1.2, Radians(20))
	left := NewMaterialSDF(
		NewTransformSDF(NewSphereSDF(0.7), Translate(V(-0.45, 0, 0))), red)
	right := NewMaterialSDF(
		NewTransformSDF(NewSphereSDF(0.7), Translate(V(0.45, 0, 0))), blue)

	// k is the blending radius; 0 (or NaN/Inf) falls back to the hard union.
	sdf := NewSmoothUnionSDF(0.6, left, right)
	scene.Add(NewSDFShape(sdf, red))

	floor := GlossyMaterial(HexColor(0xFFF0A5), 1.2, Radians(20))
	scene.Add(NewPlane(V(0, 0, -0.75), V(0, 0, 1), floor))

	camera := LookAt(V(-3, 0, 1.2), V(0, 0, 0), V(0, 0, 1), 35)
	sampler := NewSampler(4, 4)
	sampler.LightMode = LightModeAll
	sampler.SpecularMode = SpecularModeAll
	renderer := NewRenderer(&scene, &camera, sampler, 800, 800)
	renderer.IterativeRender("out%03d.png", 1000)
}
