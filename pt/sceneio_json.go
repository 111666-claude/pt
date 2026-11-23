package pt

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
)

// Stable JSON encoding for the geometric base types. Vectors and colors are
// emitted as three-component objects with fixed lowercase field order, so
// the schema does not depend on Go struct field naming. Decoding also
// accepts a concise [x, y, z] array form.

// MarshalJSON implements json.Marshaler.
func (v Vector) MarshalJSON() ([]byte, error) {
	if err := v.checkFinite(); err != nil {
		return nil, err
	}
	return []byte(fmt.Sprintf(`{"x":%s,"y":%s,"z":%s}`,
		jsonFloat(v.X), jsonFloat(v.Y), jsonFloat(v.Z))), nil
}

// UnmarshalJSON implements json.Unmarshaler. It accepts both the canonical
// {"x":..,"y":..,"z":..} object form and a compact [x, y, z] array.
func (v *Vector) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var arr [3]float64
		if err := json.Unmarshal(data, &arr); err != nil {
			return err
		}
		v.X, v.Y, v.Z = arr[0], arr[1], arr[2]
		return v.checkFinite()
	}
	var obj struct {
		X float64 `json:"x"`
		Y float64 `json:"y"`
		Z float64 `json:"z"`
	}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	v.X, v.Y, v.Z = obj.X, obj.Y, obj.Z
	return v.checkFinite()
}

func (v Vector) checkFinite() error {
	if math.IsNaN(v.X) || math.IsInf(v.X, 0) ||
		math.IsNaN(v.Y) || math.IsInf(v.Y, 0) ||
		math.IsNaN(v.Z) || math.IsInf(v.Z, 0) {
		return fmt.Errorf("vector contains a non-finite component: %v", v)
	}
	return nil
}

// MarshalJSON implements json.Marshaler.
func (c Color) MarshalJSON() ([]byte, error) {
	if math.IsNaN(c.R) || math.IsInf(c.R, 0) ||
		math.IsNaN(c.G) || math.IsInf(c.G, 0) ||
		math.IsNaN(c.B) || math.IsInf(c.B, 0) {
		return nil, fmt.Errorf("color contains a non-finite component: %v", c)
	}
	return []byte(fmt.Sprintf(`{"r":%s,"g":%s,"b":%s}`,
		jsonFloat(c.R), jsonFloat(c.G), jsonFloat(c.B))), nil
}

// UnmarshalJSON implements json.Unmarshaler. It accepts both the canonical
// {"r":..,"g":..,"b":..} object form and a compact [r, g, b] array.
func (c *Color) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var arr [3]float64
		if err := json.Unmarshal(data, &arr); err != nil {
			return err
		}
		c.R, c.G, c.B = arr[0], arr[1], arr[2]
	} else {
		var obj struct {
			R float64 `json:"r"`
			G float64 `json:"g"`
			B float64 `json:"b"`
		}
		if err := json.Unmarshal(data, &obj); err != nil {
			return err
		}
		c.R, c.G, c.B = obj.R, obj.G, obj.B
	}
	if math.IsNaN(c.R) || math.IsInf(c.R, 0) ||
		math.IsNaN(c.G) || math.IsInf(c.G, 0) ||
		math.IsNaN(c.B) || math.IsInf(c.B, 0) {
		return fmt.Errorf("color contains a non-finite component: %v", c)
	}
	return nil
}

// jsonFloat returns the exact number formatting used by encoding/json for
// float64 values (shortest round-trippable 'g' representation), so messages
// built here are byte-identical to encoder output on every platform.
func jsonFloat(x float64) string {
	return strconv.FormatFloat(x, 'g', -1, 64)
}
