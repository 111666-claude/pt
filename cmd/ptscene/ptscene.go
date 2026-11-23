// Command ptscene validates, converts and renders versioned scene
// descriptions (.ptscene.json) for fogleman/pt.
//
// Usage:
//
//	ptscene check  <scene.json>                 validate without loading assets
//	ptscene info   <scene.json>                 print a summary of the document
//	ptscene render <scene.json> <out.png> [w h] [spp]   render a preview
package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/fogleman/pt/pt"
)

func main() {
	if len(os.Args) < 3 {
		usage()
	}
	cmd, path := os.Args[1], os.Args[2]
	switch cmd {
	case "check":
		doc, err := pt.LoadSceneDocument(path, nil)
		if err != nil {
			fatal(err)
		}
		fmt.Printf("OK: schema version %d, %d materials, %d textures, %d meshes, %d shapes\n",
			doc.Version, len(doc.Materials), len(doc.Textures), len(doc.Meshes), len(doc.Shapes))
	case "info":
		doc, err := pt.LoadSceneDocument(path, nil)
		if err != nil {
			fatal(err)
		}
		printInfo(doc)
	case "render":
		if len(os.Args) < 4 {
			usage()
		}
		out := os.Args[3]
		w, h, spp := 512, 512, 8
		if len(os.Args) >= 6 {
			w, _ = strconv.Atoi(os.Args[4])
			h, _ = strconv.Atoi(os.Args[5])
		}
		if len(os.Args) >= 7 {
			spp, _ = strconv.Atoi(os.Args[6])
		}
		compiled, err := pt.LoadScene(path, nil)
		if err != nil {
			fatal(err)
		}
		if compiled.Camera == nil {
			fatal(fmt.Errorf("scene has no camera; add a \"camera\" section"))
		}
		if compiled.Sampler == nil {
			compiled.Sampler = pt.NewSampler(16, 8)
		}
		img := pt.Render(compiled.Scene, compiled.Camera, compiled.Sampler, w, h, spp)
		if err := pt.SavePNG(out, img); err != nil {
			fatal(err)
		}
		fmt.Printf("rendered %dx%d -> %s\n", w, h, out)
	default:
		usage()
	}
}

func printInfo(doc *pt.SceneDocument) {
	fmt.Printf("schema version : %d\n", doc.Version)
	if doc.Camera != nil {
		fmt.Printf("camera         : %s\n", doc.Camera.Type)
	}
	if doc.Sampler != nil {
		fmt.Printf("sampler        : %s\n", doc.Sampler.Type)
	}
	if doc.Background != nil {
		switch {
		case doc.Background.Texture != "":
			fmt.Printf("background     : texture %s\n", doc.Background.Texture)
		case doc.Background.Color != nil:
			fmt.Printf("background     : color %v\n", *doc.Background.Color)
		}
	}
	fmt.Printf("materials      : %d\n", len(doc.Materials))
	fmt.Printf("textures       : %d\n", len(doc.Textures))
	fmt.Printf("meshes         : %d\n", len(doc.Meshes))
	fmt.Printf("shapes         : %d\n", len(doc.Shapes))
	for i, s := range doc.Shapes {
		id := s.ID
		if id == "" {
			id = "-"
		}
		fmt.Printf("  [%d] %-9s id=%s material=%s\n", i, s.Type, id, s.Material)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: ptscene check|info <scene.json> | ptscene render <scene.json> <out.png> [w h] [spp]")
	os.Exit(2)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "ptscene:", err)
	os.Exit(1)
}
