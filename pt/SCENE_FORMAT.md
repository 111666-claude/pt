# Versioned Scene Description

`pt` can save and load complete scenes as versioned JSON documents
(`.ptscene.json`), so scenes no longer have to be assembled exclusively in
Go code. A document describes the camera, sampler, environment background,
materials, textures, meshes, primitive geometry and transforms.

## Quick start

```go
// load a scene, resolving textures/meshes relative to the scene file
compiled, err := pt.LoadScene("levels/level1.ptscene.json", nil)
// compiled.Scene, compiled.Camera, compiled.Sampler are ready to render

// save a scene built the classic imperative way
doc, err := pt.NewDocumentFromScene(scene, &camera, sampler, registeredTextures)
err = pt.SaveSceneDocument("levels/level1.ptscene.json", doc)
```

A command-line tool is provided for editor integration:

```sh
go run ./cmd/ptscene check  scene.ptscene.json            # validate only
go run ./cmd/ptscene info   scene.ptscene.json            # print summary
go run ./cmd/ptscene render scene.ptscene.json out.png 64 64 8
```

## Stability guarantees

- **Versioned.** Every document carries `"version"`. This build produces
  version `1` (`pt.SceneSchemaVersion`). Unknown future versions are rejected
  with an explanatory error; `pt.MigrateSceneDocument` is the extension point
  for migrations. A missing version is accepted as version 1.
- **Deterministic bytes.** Field order follows the struct order in
  `sceneio.go`, and floating-point numbers use Go's shortest round-trippable
  `strconv` format (the same one `encoding/json` uses), which is identical on
  every platform. `MarshalSceneDocument` uses two-space indentation and a
  trailing newline. Re-marshaling a loaded document yields byte-identical
  output.
- **Stable base types.** Vectors serialize as `{"x":..,"y":..,"z":..}` and
  colors as `{"r":..,"g":..,"b":..}`; both also accept a compact
  `[x, y, z]` / `[r, g, b]` array on input. Matrices are flat 16-number
  arrays in row-major order.
- **Unknown fields ignored.** Readers skip fields they do not recognize, so
  writers can extend the schema without breaking older renderers.
- **Defaults preserved.** Optional scalar fields use pointers internally:
  an omitted field stays omitted on the wire and receives its documented
  default only when the live object is constructed.

## Relative resource roots

Texture and mesh paths must be **relative, slash-separated paths inside the
scene's root directory**. Absolute paths, drive letters, backslashes and
`..` traversal that escapes the root are rejected. The root defaults to the
directory containing the scene file and can be overridden via
`LoadOptions.Root`.

```json
{ "id": "tex-wood", "path": "assets/wood.png" }
```

## Document outline

```jsonc
{
  "version": 1,
  "camera":     { /* CameraDocument */ },
  "sampler":    { /* SamplerDocument */ },
  "background": { /* color or environment texture */ },
  "materials":  [ /* MaterialDocument, referenced by id */ ],
  "textures":   [ /* TextureDocument, referenced by id */ ],
  "meshes":     [ /* file or inline triangle meshes, referenced by id */ ],
  "shapes":     [ /* primitives, meshes, references and transforms */ ]
}
```

### Camera

Two input forms are accepted; output always uses the exact `basis` form so
round-tripping is bit-for-bit.

```json
{ "type": "basis",
  "origin":  {"x":3,"y":2,"z":2},
  "right":   {"x":1,"y":0,"z":0},
  "up":      {"x":0,"y":1,"z":0},
  "forward": {"x":0,"y":0,"z":-1},
  "m": 2.7474774194546225,
  "focus": { "focalPoint": {"x":0,"y":0,"z":0}, "apertureRadius": 0.01 } }
```

The ergonomic form is converted to the basis form on load:

```json
{ "type": "lookAt", "eye": [3,2,2], "center": [0,0,0],
  "worldUp": [0,1,0], "fovy": 40 }
```

### Sampler

All fields are optional; shown values are the construction defaults.

```json
{ "type": "default",
  "firstHitSamples": 16, "maxBounces": 8,
  "directLighting": true, "softShadows": true,
  "lightMode": "random",      // random | all
  "specularMode": "naive" }   // naive | first | all
```

### Background

Exactly one of `color` or `texture` (a texture id) is required.

```json
{ "color": {"r":0.05,"g":0.07,"b":0.1} }
{ "texture": "tex-env", "textureAngle": 0.5 }
```

### Materials

All numeric/boolean fields are optional and default to the values used by
`pt.DiffuseMaterial(Black)`. Texture slots reference texture ids.

```json
{ "id": "mat-floor",
  "color": {"r":0.8,"g":0.8,"b":0.7},
  "texture": "tex-checker",
  "normalTexture": "", "bumpTexture": "", "glossTexture": "",
  "bumpMultiplier": 1, "emittance": 0, "index": 1,
  "gloss": 0, "tint": 0, "reflectivity": -1, "transparent": false }
```

### Textures

```json
{ "id": "tex-checker", "path": "checker.png",
  "pow": 0.45454545454545453, "scale": 1.0 }
```

`pow` and `scale` are optional post-decode adjustments, matching
`NewTexture(img).Pow(pow).MulScalar(scale)`.

### Meshes

A mesh either references an OBJ/STL file or carries inline triangles
(`format` is inferred from the extension when omitted).

```json
{ "id": "mesh-file", "path": "teapot.obj", "material": "mat-floor",
  "smoothNormals": true }
```

```json
{ "id": "mesh-inline", "triangles": [
  { "v1": [0,0,0], "v2": [1,0,0], "v3": [0,1,0],
    "n1": [0,0,1], "n2": [0,0,1], "n3": [0,0,1],
    "t1": [0,0,0], "t2": [1,0,0], "t3": [0,1,0],
    "material": "mat-floor" } ] }
```

### Shapes

Supported `type` values: `sphere`, `plane`, `cube`, `cylinder`, `mesh`,
`reference`. Every shape may carry an optional `id`, a `material` id and an
optional transform `matrix`.

```json
{ "id": "ball", "type": "sphere", "material": "mat-floor",
  "center": {"x":0,"y":0,"z":1}, "radius": 1 }

{ "type": "plane", "material": "mat-floor",
  "point": [0,0,0], "normal": [0,0,1] }

{ "type": "cube", "material": "mat-floor",
  "min": [-2,-2,1], "max": [-1,-1,2] }

{ "type": "cylinder", "material": "mat-floor",
  "radius": 0.5, "z0": 0, "z1": 2 }

{ "type": "mesh", "material": "mat-floor", "mesh": "mesh-file",
  "matrix": [ /* 16 row-major numbers */ ] }

{ "type": "reference", "ref": "ball",
  "matrix": [ /* optional: wrap the target in a transform */ ] }
```

A `reference` reuses another shape by id, which makes instancing cheap.
References may chain but must not form a cycle; the graph is checked for
cycles and for a maximum reference depth.

## Validation and error handling

Errors are returned as `*pt.SchemaError` values carrying the JSON path of
the offending element, for example:

```
$.shapes[3].matrix: matrix is singular (determinant 0), it cannot be inverted
```

Loading rejects, with contextual errors:

- missing or unreadable texture/mesh files and directories given as files;
- absolute paths, backslashes and root-escape traversal;
- duplicate material, texture, mesh and shape ids and dangling references;
- circular or excessively deep shape references;
- singular, non-invertible or non-finite transform matrices;
- non-finite (`NaN`/`Inf`) numbers in vectors, colors and scalars;
- degenerate geometry (zero normals, repeated vertices, inverted boxes,
  non-positive radii, `z1 <= z0` cylinders);
- documents or arrays exceeding the configurable `LoadOptions` limits.

A failed load returns a nil `CompiledScene`, never a partially built scene,
and does not touch the package-global texture cache. `SaveSceneDocument`
writes through a temp file and renames atomically, so a validation failure
or write error cannot truncate an existing scene file.
