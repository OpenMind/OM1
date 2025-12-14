import * as THREE from 'three'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, StatsGl } from '@react-three/drei'
import { useEffect, useRef, useState } from 'react'

const BRIDGE = (import.meta.env.VITE_BRIDGE_WS || 'ws://localhost:8081/ws')
  .replace(/^ws(s)?:\/\//, 'http$1://').replace(/\/ws$/, '')

function Bot({ pose }) {
  return (
    <group position={[pose.x, 0.2, pose.z]} rotation={[0, pose.yaw, 0]}>
      <mesh castShadow>
        <boxGeometry args={[0.4, 0.2, 0.6]} />
        <meshStandardMaterial />
      </mesh>
    </group>
  )
}

function Obstacles(){
  return (
    <group>
      <mesh position={[ 1.2,0.25, 1.2]} castShadow receiveShadow><boxGeometry args={[0.5,0.5,0.5]} /><meshStandardMaterial color="#888" /></mesh>
      <mesh position={[-1.0,0.25, 1.8]} castShadow receiveShadow><boxGeometry args={[0.5,0.5,0.5]} /><meshStandardMaterial color="#888" /></mesh>
      <mesh position={[ 0.2,0.25,-1.2]} castShadow receiveShadow><boxGeometry args={[0.5,0.5,0.5]} /><meshStandardMaterial color="#888" /></mesh>
    </group>
  )
}

function SensorLines({ pose, sensors }){
  const groupRef = useRef()
  const linesRef = useRef([])

  useEffect(()=>{
    if (!groupRef.current) return
    while (groupRef.current.children.length) groupRef.current.remove(groupRef.current.children[0])
    linesRef.current = []
    for (let i=0;i<(sensors?.beams||0);i++){
      const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()])
      const mat  = new THREE.LineBasicMaterial({ transparent:true, opacity:0.9 })
      const line = new THREE.Line(geom, mat)
      groupRef.current.add(line)
      linesRef.current.push(line)
    }
  }, [sensors?.beams])

  useEffect(()=>{
    if (!sensors || linesRef.current.length===0) return
    const { maxRange, fov, beams, distances } = sensors
    const origin = new THREE.Vector3(pose.x, 0.2, pose.z)
    const start = -fov/2, step = fov/(beams-1)
    for (let i=0;i<beams;i++){
      const ang = pose.yaw + start + step*i
      const dir = new THREE.Vector3(Math.sin(ang), 0, Math.cos(ang))
      const dist = distances[i] ?? maxRange
      const line = linesRef.current[i]
      if (!line) continue
      line.geometry.setFromPoints([origin, origin.clone().add(dir.multiplyScalar(dist))])
      const t = dist/maxRange
      line.material.color.setHSL(0.66*(1-t), 1, 0.5) // dekat=merah, jauh=biru
    }
  }, [pose, sensors])

  return <group ref={groupRef} />
}

async function step(v,w){
  await fetch(`${BRIDGE}/action`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ v,w }) })
}
async function resetSim(){ await fetch(`${BRIDGE}/reset`, { method:'POST' }) }

export default function App(){
  const [pose, setPose] = useState({x:0,z:0,yaw:0})
  const [sensors, setSensors] = useState(null)
  const [hud, setHud] = useState({minDist: null, reward: 0, steps: 0, collisions: 0, done: false})

  useEffect(()=>{
    const url = import.meta.env.VITE_BRIDGE_WS || 'ws://localhost:8081/ws'
    const ws = new WebSocket(url)
    ws.onmessage = (ev)=>{
      const msg = JSON.parse(ev.data)
      if (msg.type==='state'){
        setPose(msg.pose)
        setSensors(msg.sensors)
        const minDist = msg.sensors ? Math.min(...msg.sensors.distances) : null
        setHud(h => ({ ...h, minDist, reward: msg.reward ?? 0, done: !!msg.done }))
      }
    }
    const onKey = (e)=>{
      if (e.repeat) return
      const V=0.06, W=0.12
      if (e.key==='w'||e.key==='ArrowUp') step(V,0)
      if (e.key==='s'||e.key==='ArrowDown') step(-V,0)
      if (e.key==='a'||e.key==='ArrowLeft') step(0,+W)
      if (e.key==='d'||e.key==='ArrowRight') step(0,-W)
      if (e.key==='r') resetSim()
    }
    window.addEventListener('keydown', onKey)
    return ()=>{ ws.close(); window.removeEventListener('keydown', onKey) }
  },[])

  return (
    <>
      <Canvas shadows camera={{ position:[2,2,2] }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[3,5,2]} intensity={0.8} castShadow />
        <Grid args={[10,10]} />
        <Obstacles />
        <Bot pose={pose} />
        {sensors && <SensorLines pose={pose} sensors={sensors} />}
        <OrbitControls />
        <StatsGl />
      </Canvas>

      {/* HUD */}
      <div style={{position:'fixed', left:16, bottom:16, padding:8, background:'#0008', color:'#fff', borderRadius:8}}>
        <div>minDist: {hud.minDist?.toFixed ? hud.minDist.toFixed(2) : '–'} m</div>
        <div>reward: {hud.reward?.toFixed ? hud.reward.toFixed(3) : '0.000'}</div>
        <div>done: {String(hud.done)}</div>
        <div style={{marginTop:8, display:'flex', gap:8}}>
          <button onClick={()=>step(0.06,0)} style={{padding:8}}>Forward</button>
          <button onClick={()=>step(0,+0.12)} style={{padding:8}}>Left</button>
          <button onClick={()=>step(0,-0.12)} style={{padding:8}}>Right</button>
          <button onClick={()=>step(-0.06,0)} style={{padding:8}}>Back</button>
          <button onClick={()=>resetSim()} style={{padding:8}}>Reset (R)</button>
        </div>
      </div>
    </>
  )
}
