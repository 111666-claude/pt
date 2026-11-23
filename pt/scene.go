package pt

import (
	"math"
	"sync/atomic"
)

type Scene struct {
	Color        Color
	Texture      Texture
	TextureAngle float64
	Shapes       []Shape
	Lights       []Shape
	tree         *Tree
	media        []*Volume
	rays         uint64
}

func (s *Scene) Compile() {
	for _, shape := range s.Shapes {
		shape.Compile()
	}
	s.media = s.media[:0]
	for _, shape := range s.Shapes {
		if v, ok := shape.(*Volume); ok && v.IsMedium() {
			s.media = append(s.media, v)
		}
	}
	if s.tree == nil {
		s.tree = NewTree(s.Shapes)
	}
}

// HasMedia reports whether any shape in the scene is a participating medium.
func (s *Scene) HasMedia() bool {
	return len(s.media) > 0
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

// Transmittance accumulates beam transmittance through every participating
// medium intersected by the ray over [tmin, tmax]. This is the single
// density-field entry point shared by shadow, reflection and refraction
// rays, so all ray classes observe identical optical thickness. Scenes
// without media return White and incur no cost.
func (s *Scene) Transmittance(ray Ray, tmin, tmax float64) Color {
	if len(s.media) == 0 {
		return White
	}
	if !isFinite(tmin) || !isFinite(tmax) || tmax <= tmin {
		return White
	}
	if tmin < 0 {
		tmin = 0
	}
	result := White
	for _, v := range s.media {
		b0, b1 := v.Box.Intersect(ray)
		if !isFinite(b0) || !isFinite(b1) || b1 <= b0 || b1 <= 0 {
			continue
		}
		lo := math.Max(b0, tmin)
		hi := math.Min(b1, tmax)
		if hi <= lo {
			continue
		}
		result = result.Mul(v.Transmittance(ray, lo, hi))
	}
	return result
}
