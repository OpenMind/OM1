package memory

import (
	"strings"

	"github.com/openmind/om1/internal/providers"
)

func ResolveCurrentUser() string {
	face := providers.IO().GetInput("FacePresence")
	if face == nil || face.Input == "" {
		return ""
	}

	const marker = "Closest: "
	idx := strings.Index(face.Input, marker)
	if idx < 0 {
		return ""
	}

	name := face.Input[idx+len(marker):]
	name = strings.TrimRight(name, ".")
	name = strings.TrimSpace(name)
	name = strings.ToLower(name)

	if name == "" || name == "unknown" {
		return ""
	}
	return name
}
