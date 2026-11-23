package pt

type Material struct {
	Color          Color
	Texture        Texture
	NormalTexture  Texture
	BumpTexture    Texture
	GlossTexture   Texture
	BumpMultiplier float64
	Emittance      float64
	Index          float64 // refractive index
	Gloss          float64 // reflection cone angle in radians
	Tint           float64 // specular and refractive tinting
	Reflectivity   float64 // metallic reflection
	Transparent    bool
	// Roughness is the linear microfacet roughness in [0, 1]. A negative
	// value (the default) disables the GGX path and keeps the legacy Gloss
	// cone model. Zero is a stable perfect specular mirror.
	Roughness float64
	// Metalness blends between a dielectric Fresnel derived from Index (0)
	// and a conductor whose base reflectance is the material Color (1).
	Metalness float64
}

func DiffuseMaterial(color Color) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, 1, 0, 0, -1, false, -1, 0}
}

func SpecularMaterial(color Color, index float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, index, 0, 0, -1, false, -1, 0}
}

func GlossyMaterial(color Color, index, gloss float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, index, gloss, 0, -1, false, -1, 0}
}

func ClearMaterial(index, gloss float64) Material {
	return Material{Black, nil, nil, nil, nil, 1, 0, index, gloss, 0, -1, true, -1, 0}
}

func TransparentMaterial(color Color, index, gloss, tint float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, index, gloss, tint, -1, true, -1, 0}
}

func MetallicMaterial(color Color, gloss, tint float64) Material {
	return Material{color, nil, nil, nil, nil, 1, 0, 1, gloss, tint, 1, false, -1, 0}
}

func LightMaterial(color Color, emittance float64) Material {
	return Material{color, nil, nil, nil, nil, 1, emittance, 1, 0, 0, -1, false, -1, 0}
}

// GGXMaterial creates a microfacet material with the GGX normal distribution
// and Smith masking term. roughness is the linear roughness in [0, 1]
// (0 = perfect mirror) and metalness selects dielectric (0) or metallic (1)
// Fresnel behavior. Dielectrics use index as the refractive index; metals
// use color as their normal-incidence reflectance.
func GGXMaterial(color Color, index, roughness, metalness float64) Material {
	m := Material{
		Color:          color,
		BumpMultiplier: 1,
		Index:          index,
		Reflectivity:   -1, // Fresnel is computed from index / metalness
		Roughness:      roughness,
		Metalness:      metalness,
	}
	return m.Validate()
}

// MetalMaterial is a fully metallic GGX material (metalness = 1). The color
// acts as the conductor's base reflectance.
func MetalMaterial(color Color, roughness float64) Material {
	return GGXMaterial(color, 1, roughness, 1)
}

// DielectricMaterial is a non-metallic GGX material (metalness = 0) whose
// Fresnel reflectance is derived from the refractive index.
func DielectricMaterial(color Color, index, roughness float64) Material {
	return GGXMaterial(color, index, roughness, 0)
}

// Validate clamps all parameters into valid ranges and replaces non-finite
// values with safe defaults, guaranteeing finite, non-negative sampling
// weights. It is called automatically while shading; users may call it to
// normalize hand-built Material values as well.
func (m Material) Validate() Material {
	return sanitizeMaterial(m)
}

// MaterialAt returns the material at a point on the shape, applying texture
// overrides and enforcing valid parameter ranges before shading.
func MaterialAt(shape Shape, point Vector) Material {
	material := shape.MaterialAt(point)
	uv := shape.UV(point)
	if material.Texture != nil {
		material.Color = material.Texture.Sample(uv.X, uv.Y)
	}
	if material.GlossTexture != nil {
		c := material.GlossTexture.Sample(uv.X, uv.Y)
		material.Gloss = (c.R + c.G + c.B) / 3
	}
	material = sanitizeMaterial(material)
	// A microfacet material tints its reflection by the material color when
	// metallic, mirroring MetallicMaterial's tint == 1 convention.
	if material.ggxEnabled() && material.Metalness > 0 && material.Tint <= 0 {
		material.Tint = material.Metalness
	}
	return material
}
