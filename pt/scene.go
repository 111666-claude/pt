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
}

func (s *Scene) Compile() {
	for _, shape := range s.Shapes {
		shape.Compile()
	}
	if s.tree == nil {
		s.tree = NewTree(s.Shapes)
	}
}

func (s *Scene) Add(shape Shape) {
	s.Shapes = append(s.Shapes, shape)
	// area lights decide themselves whether they carry any emissive
	// surface: Mesh.MaterialAt is unimplemented, so emittance must be
	// queried through the area distribution instead of MaterialAt.
	if area, ok := shape.(AreaLight); ok {
		if area.LightArea() > 0 {
			s.Lights = append(s.Lights, shape)
		}
		return
	}
	// other shapes (sphere, cube fallback, plane, ...): only positive,
	// finite emittance makes them a light. Negative emittance has no
	// physical meaning and is treated as non-emissive.
	if shapeEmittance(shape) > 0 {
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
