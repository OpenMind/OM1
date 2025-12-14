const express = require('express')
const cors = require('cors')
const { WebSocketServer } = require('ws')

const app = express()
app.use(cors()); app.use(express.json())

// --- world state ---
let pose = { x:0, z:0, yaw:0 }
let steps = 0, collisions = 0
let lastReward = 0, lastDone = false

// Obstacles (AABB di bidang XZ)
const boxes = [
  { cx:  1.2, cz:  1.2, w:0.5, h:0.5 },
  { cx: -1.0, cz:  1.8, w:0.5, h:0.5 },
  { cx:  0.2, cz: -1.2, w:0.5, h:0.5 },
].map(b => ({ min:{x:b.cx-b.w/2, z:b.cz-b.h/2}, max:{x:b.cx+b.w/2, z:b.cz+b.h/2} }))

// Sensor config
const S = { maxRange:4, fov:Math.PI/2, beams:13 }

// Ray (o,d) vs AABB 2D
function rayAabb2D(o, d, box){
  const invDx = 1/(d.x || 1e-9), invDz = 1/(d.z || 1e-9)
  let t1 = (box.min.x - o.x) * invDx, t2 = (box.max.x - o.x) * invDx
  let t3 = (box.min.z - o.z) * invDz, t4 = (box.max.z - o.z) * invDz
  const tmin = Math.max(Math.min(t1,t2), Math.min(t3,t4))
  const tmax = Math.min(Math.max(t1,t2), Math.max(t3,t4))
  if (tmax < 0) return Infinity
  if (tmin > tmax) return Infinity
  const t = tmin >= 0 ? tmin : tmax
  return t >= 0 ? t : Infinity
}

function computeSensors(){
  const distances = []
  const start = -S.fov/2, step = S.fov/(S.beams-1)
  for (let i=0;i<S.beams;i++){
    const ang = pose.yaw + start + step*i
    const dir = { x: Math.sin(ang), z: Math.cos(ang) }
    let tHit = Infinity
    for (const b of boxes){
      const t = rayAabb2D(pose, dir, b)
      if (t < tHit) tHit = t
    }
    distances.push(Math.min(S.maxRange, tHit))
  }
  return { ...S, distances }
}

function applyAction(v=0.05, w=0){
  const prev = { ...pose }
  pose.yaw += w
  pose.x   += v * Math.sin(pose.yaw)
  pose.z   += v * Math.cos(pose.yaw)

  const sensors = computeSensors()
  const minDist = Math.min(...sensors.distances)
  const collided = minDist < 0.25

  if (collided){
    // rollback posisi jika tabrakan
    pose = prev
    collisions += 1
  }

  // reward sederhana: maju +, mundur -, belok ada penalti kecil, tabrakan penalti besar
  const reward = (v >= 0 ? +Math.abs(v) : -0.5*Math.abs(v)) - 0.05*Math.abs(w) - (collided ? 1.0 : 0)
  steps += 1
  const done = collisions >= 5 || steps >= 1000

  lastReward = reward
  lastDone = done

  return { sensors, reward, done, info: { steps, collisions, minDist } }
}

// --- endpoints ---
app.post('/action', (req, res) => {
  const { v=0.05, w=0 } = req.body || {}
  const result = applyAction(v, w)
  res.json({ ok:true, ...result })
})

app.post('/reset', (req, res) => {
  pose = { x:0, z:0, yaw:0 }
  steps = 0; collisions = 0; lastReward = 0; lastDone = false
  res.json({ ok:true })
})

// --- server ---
const server = app.listen(8081, '0.0.0.0', () => console.log('bridge :8081'))
const wss = new WebSocketServer({ server, path:'/ws' })
wss.on('connection', (ws) => {
  ws.send(JSON.stringify({ type:'hello' }))
  const itv = setInterval(()=>{
    ws.send(JSON.stringify({ type:'state', pose, sensors: computeSensors(), reward:lastReward, done:lastDone }))
  }, 50)
  ws.on('close', () => clearInterval(itv))
})
