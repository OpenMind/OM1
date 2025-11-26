#!/bin/bash
# Spawn OM1 robot in Gazebo

echo "Spawning OM1 robot in Gazebo..."

gz service -s /world/office_world/create \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req "sdf_filename: 'models/enhanced_robot/model.sdf', name: 'om1_robot', pose: {position: {x: 0, y: 0, z: 0.3}}"

echo "Robot spawned successfully!"
