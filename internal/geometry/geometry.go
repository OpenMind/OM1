package geometry

// Point is a geometry_msgs/Point: a 3D position.
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

// Quaternion is a geometry_msgs/Quaternion: an orientation.
type Quaternion struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
	W float64 `json:"w"`
}

// Pose is a geometry_msgs/Pose: a position paired with an orientation.
type Pose struct {
	Position    Point      `json:"position"`
	Orientation Quaternion `json:"orientation"`
}
