package memory

import (
	"github.com/openmind/om1/internal/providers"
)

// UserIdentity holds the resolved UUID and display name for the current user.
type UserIdentity struct {
	UUID string
	Name string
}

// ResolveCurrentUser reads the face-presence dynamic vars set by the sensor.
func ResolveCurrentUser() UserIdentity {
	io := providers.IO()
	uuid, _ := io.GetDynamicVar("current_user_uuid")
	name, _ := io.GetDynamicVar("current_user_name")
	return UserIdentity{UUID: uuid, Name: name}
}
