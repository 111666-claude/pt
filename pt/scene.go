package pt

import "sync/atomic"

type Scene struct {
	Color        Color
	Texture      Texture
	TextureAngle float64
	Shapes       []Shape
	Lights       []Shape
	tree         *Tree
	rays         uint64
	// EnvSampling turns on environment map importance sampling. It defaults
	// to false so existing renders keep their original path behavior.
	EnvSampling bool
	// envDistribution is built once (lazily or in Compile) and then shared
	// read-only by every rendering goroutine.
	envDistribution *EnvDistribution
	envCompiled     bool
}

func (s *Scene) Compile() {
	for _, shape := range s.Shapes {
		shape.Compile()
	}
	if s.tree == nil {
		s.tree = NewTree(s.Shapes)
	}
	if s.EnvSampling && !s.envCompiled {
		s.BuildEnvDistribution()
	}
}

func (s *Scene) Add(shape Shape) {
	s.Shapes = append(s.Shapes, shape)
	if shape.MaterialAt(Vector{}).Emittance > 0 {
		s.Lights = append(s.Lights, shape)
	}
}

func (s *Scene) RayCount() uint64 {
	return atomic.LoadUint64(&s.rays)
}

func (s *Scene) Intersect(r Ray) Hit {
	atomic.AddUint64(&s.rays, 1)
	return s.tree.Intersect(r)
}

// BuildEnvDistribution (re)builds the reusable environment importance
// sampling distribution from the current scene texture and rotation. It is
// safe to call explicitly; Compile calls it automatically when EnvSampling
// is enabled. A nil/black/degenerate texture yields an invalid distribution
// whose Valid() method reports false, in which case sampling silently falls
// back to the plain environment color path without any division by zero.
func (s *Scene) BuildEnvDistribution() {
	if s.Texture == nil {
		s.envDistribution = &EnvDistribution{}
	} else {
		s.envDistribution = NewEnvDistribution(s.Texture, s.TextureAngle)
	}
	s.envCompiled = true
}

// EnvDistribution returns the cached environment distribution, building it
// on first use. The returned pointer must be treated as read-only.
func (s *Scene) EnvDistribution() *EnvDistribution {
	if !s.envCompiled {
		s.BuildEnvDistribution()
	}
	return s.envDistribution
}
