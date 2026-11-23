package pt

import (
	"image"
	"math"
)

type Channel int

const (
	ColorChannel = iota
	VarianceChannel
	StandardDeviationChannel
	SamplesChannel
)

type Pixel struct {
	Samples int
	M, V    Color
}

func (p *Pixel) AddSample(sample Color) {
	p.Samples++
	if p.Samples == 1 {
		p.M = sample
		return
	}
	m := p.M
	p.M = p.M.Add(sample.Sub(p.M).DivScalar(float64(p.Samples)))
	p.V = p.V.Add(sample.Sub(m).Mul(sample.Sub(p.M)))
}

func (p *Pixel) Color() Color {
	return p.M
}

func (p *Pixel) Variance() Color {
	if p.Samples < 2 {
		return Black
	}
	return p.V.DivScalar(float64(p.Samples - 1))
}

func (p *Pixel) StandardDeviation() Color {
	return p.Variance().Pow(0.5)
}

type Buffer struct {
	W, H   int
	Pixels []Pixel
}

func NewBuffer(w, h int) *Buffer {
	pixels := make([]Pixel, w*h)
	return &Buffer{w, h, pixels}
}

func (b *Buffer) Copy() *Buffer {
	pixels := make([]Pixel, b.W*b.H)
	copy(pixels, b.Pixels)
	return &Buffer{b.W, b.H, pixels}
}

func (b *Buffer) AddSample(x, y int, sample Color) {
	b.Pixels[y*b.W+x].AddSample(sample)
}

func (b *Buffer) Samples(x, y int) int {
	return b.Pixels[y*b.W+x].Samples
}

func (b *Buffer) Color(x, y int) Color {
	return b.Pixels[y*b.W+x].Color()
}

func (b *Buffer) Variance(x, y int) Color {
	return b.Pixels[y*b.W+x].Variance()
}

func (b *Buffer) StandardDeviation(x, y int) Color {
	return b.Pixels[y*b.W+x].StandardDeviation()
}

func (b *Buffer) Image(channel Channel) image.Image {
	return b.ImageWithToneMapping(channel, NewToneMapping())
}

// ImageWithToneMapping renders the requested channel as an image. Tone
// mapping only affects the ColorChannel; Variance, StandardDeviation and
// Samples channels are produced exactly as in Image.
//
// The buffer's accumulated pixel data is never modified. Invalid tone
// mapping parameters are replaced with the gamma defaults and any non-finite
// pixel values are sanitized, so the resulting image never contains NaN or
// Inf values.
func (b *Buffer) ImageWithToneMapping(channel Channel, mapping ToneMapping) image.Image {
	mapping = mapping.normalized()
	result := image.NewRGBA64(image.Rect(0, 0, b.W, b.H))
	var maxSamples float64
	if channel == SamplesChannel {
		for _, pixel := range b.Pixels {
			maxSamples = math.Max(maxSamples, float64(pixel.Samples))
		}
	}
	for y := 0; y < b.H; y++ {
		for x := 0; x < b.W; x++ {
			var c Color
			switch channel {
			case ColorChannel:
				c = mapping.apply(b.Pixels[y*b.W+x].Color())
			case VarianceChannel:
				c = b.Pixels[y*b.W+x].Variance()
			case StandardDeviationChannel:
				c = b.Pixels[y*b.W+x].StandardDeviation()
			case SamplesChannel:
				p := 0.0
				if maxSamples > 0 {
					p = float64(b.Pixels[y*b.W+x].Samples) / maxSamples
				}
				c = Color{p, p, p}
			}
			result.SetRGBA64(x, y, c.RGBA64())
		}
	}
	return result
}
