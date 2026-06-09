package autonomy

import (
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

// serializeTwist encodes a geometry_msgs/Twist message in CDR little-endian
// format, ready to publish on a ROS2 /cmd_vel topic.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  linear.x   float64 LE
//	[12] linear.y   float64 LE
//	[20] linear.z   float64 LE
//	[28] angular.x  float64 LE
//	[36] angular.y  float64 LE
//	[44] angular.z  float64 LE
//
// Every field is a float64 starting on an 8-byte aligned data offset, so no
// interior padding is required.
func serializeTwist(linearX, linearY, linearZ, angularX, angularY, angularZ float64) []byte {
	buf := make([]byte, 0, 4+6*8)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendFloat64LE(buf, linearX)
	buf = zenohsession.AppendFloat64LE(buf, linearY)
	buf = zenohsession.AppendFloat64LE(buf, linearZ)
	buf = zenohsession.AppendFloat64LE(buf, angularX)
	buf = zenohsession.AppendFloat64LE(buf, angularY)
	buf = zenohsession.AppendFloat64LE(buf, angularZ)

	return buf
}
