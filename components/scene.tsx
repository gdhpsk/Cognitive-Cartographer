'use client';

import { useEffect, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import gsap from 'gsap';
import { useAppStore, type NodeData, type EdgeData } from '@/helpers/store';
import * as THREE from 'three';
import { forceSimulation, forceLink, forceManyBody, forceCenter, forceRadial, type SimulationNode, type SimulationLink } from 'd3-force-3d';

interface SimNode extends SimulationNode {
  id: string;
  label: string;
  color: string;
}

interface SimLink extends SimulationLink<SimNode> {
  source: SimNode | string;
  target: SimNode | string;
}

const white = new THREE.Color('white');
const red = new THREE.Color('#dbf702');
const blue = new THREE.Color('#96b2ff');
const targetColor = new THREE.Color();
const SOFT_BOUNDARY_RADIUS = 7.4;
const OVERFLOW_PULL = 0.18;
const CYLINDER_BASE_DIR = new THREE.Vector3(0, 1, 0);
const SELECTED_EDGE_RADIUS = 0.02;

function ForceGraph({ nodes, edges, simNodesRefOut }: { nodes: NodeData[]; edges: EdgeData[]; simNodesRefOut?: React.RefObject<SimNode[]> }) {
  const meshRefs = useRef<Map<string, THREE.Mesh>>(new Map());
  const simRef = useRef<ReturnType<typeof forceSimulation<SimNode>> | null>(null);
  const simNodesRef = useRef<SimNode[]>([]);
  const simLinksRef = useRef<SimLink[]>([]);
  const lineGeoRef = useRef<THREE.BufferGeometry>(null);
  const activeLineGeoRef = useRef<THREE.BufferGeometry>(null);
  const selectedLineGeoRef = useRef<THREE.BufferGeometry>(null);
  const selectedEdgeMeshRef = useRef<THREE.InstancedMesh>(null);
  const selectedEdgeDummyRef = useRef<THREE.Object3D>(new THREE.Object3D());
  const tooltipRef = useRef<THREE.Group>(null);
  const tooltipLabelRef = useRef<any>(null);
  const tooltipCoordsRef = useRef<any>(null);

  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Create / recreate simulation when nodes or edges change
  useEffect(() => {
    if (nodes.length === 0) {
      simRef.current = null;
      simNodesRef.current = [];
      simLinksRef.current = [];
      return;
    }

    const simNodes: SimNode[] = nodes.map((n) => ({
      id: n.id,
      label: n.label,
      color: n.color,
      x: n.position[0],
      y: n.position[1],
      z: n.position[2],
    }));

    const simLinks: SimLink[] = edges.map((e) => ({
      source: e.sourceId,
      target: e.targetId,
    }));

    simNodesRef.current = simNodes;
    simLinksRef.current = simLinks;
    if (simNodesRefOut) (simNodesRefOut as { current: SimNode[] }).current = simNodes;

    // Preallocate edge position buffer
    if (lineGeoRef.current) {
      const positions = new Float32Array(Math.max(edges.length * 6, 6));
      lineGeoRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      lineGeoRef.current.setDrawRange(0, edges.length * 2);
    }

    // Preallocate active edge position buffer
    if (activeLineGeoRef.current) {
      const positions = new Float32Array(Math.max(edges.length * 6, 6));
      activeLineGeoRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      activeLineGeoRef.current.setDrawRange(0, 0);
    }

    // Preallocate selected edge position buffer
    if (selectedLineGeoRef.current) {
      const positions = new Float32Array(Math.max(edges.length * 6, 6));
      selectedLineGeoRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      selectedLineGeoRef.current.setDrawRange(0, 0);
    }

    // IMPORTANT: pass numDimensions as 2nd arg — constructor initializes nodes
    // before chained methods run, so .numDimensions(3) after construction is too late
    const sim = forceSimulation<SimNode>(simNodes, 3)
      .force('link', forceLink<SimNode, SimLink>(simLinks).id((d: SimNode) => d.id).distance(1.2))
      .force('charge', forceManyBody<SimNode>().strength(-0.9).distanceMax(4))
      .force('center', forceCenter<SimNode>(0, 0, 0))
      .force('compactness', forceRadial<SimNode>(0).strength(0.06))
      .alphaDecay(0.02)
      .velocityDecay(0.3);

    // Stop the auto-timer immediately — we tick manually in useFrame
    sim.stop();
    simRef.current = sim;
  }, [nodes, edges]);

  // Each frame: tick simulation, move meshes, update edge buffer, update tooltip
  useFrame(() => {
    const sim = simRef.current;
    const simNodes = simNodesRef.current;
    const simLinks = simLinksRef.current;
    const lineGeo = lineGeoRef.current;
    const activeLineGeo = activeLineGeoRef.current;
    const selectedLineGeo = selectedLineGeoRef.current;

    if (!sim || simNodes.length === 0) return;

    sim.tick();

    // Softly nudge overflow nodes inward to keep the graph mostly inside the sphere.
    for (const sn of simNodes) {
      const x = sn.x ?? 0;
      const y = sn.y ?? 0;
      const z = sn.z ?? 0;
      const mag = Math.sqrt(x * x + y * y + z * z);
      if (mag > SOFT_BOUNDARY_RADIUS && mag > 0) {
        const nx = x / mag;
        const ny = y / mag;
        const nz = z / mag;
        const overflow = mag - SOFT_BOUNDARY_RADIUS;
        sn.vx = (sn.vx ?? 0) - nx * overflow * OVERFLOW_PULL;
        sn.vy = (sn.vy ?? 0) - ny * overflow * OVERFLOW_PULL;
        sn.vz = (sn.vz ?? 0) - nz * overflow * OVERFLOW_PULL;
      }
    }

    const { activeNodeIds, selectedNodeId } = useAppStore.getState();
    const selectedClusterIds = new Set<string>();
    if (selectedNodeId) {
      selectedClusterIds.add(selectedNodeId);
      for (const edge of edges) {
        if (edge.sourceId === selectedNodeId) selectedClusterIds.add(edge.targetId);
        if (edge.targetId === selectedNodeId) selectedClusterIds.add(edge.sourceId);
      }
    }

    // Move each mesh to its simulation position and lerp colors
    for (const sn of simNodes) {
      const mesh = meshRefs.current.get(sn.id);
      if (mesh) {
        mesh.position.set(sn.x ?? 0, sn.y ?? 0, sn.z ?? 0);

        mesh.scale.setScalar(1);

        const mat = mesh.material as THREE.MeshStandardMaterial;
        const isActive = activeNodeIds.has(sn.id);
        const isSelectedNode = selectedNodeId === sn.id;
        const isSelectedCluster = selectedClusterIds.has(sn.id);

        if (isSelectedNode) {
          targetColor.copy(blue);
          mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, 1.2, 0.08);
          mat.opacity = 1;
        } else if (isSelectedCluster) {
          targetColor.copy(red);
          mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, 2.3, 0.08);
          mat.opacity = 1;
        } else if (isActive) {
          targetColor.copy(white);
          mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, 2.0, 0.05);
          mat.opacity = THREE.MathUtils.lerp(mat.opacity, 1, 0.08);
        } else {
          targetColor.set(sn.color);
          mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, 0.8, 0.05);
          mat.opacity = THREE.MathUtils.lerp(mat.opacity, 0.5, 0.08);
        }

        mat.transparent = true;
        mat.color.lerp(targetColor, 0.05);
        mat.emissive.lerp(targetColor, 0.05);
      }
    }

    // Update all edge lines
    if (lineGeo) {
      const posAttr = lineGeo.getAttribute('position') as THREE.BufferAttribute | undefined;
      if (posAttr && posAttr.array.length >= simLinks.length * 6) {
        const arr = posAttr.array as Float32Array;
        for (let i = 0; i < simLinks.length; i++) {
          const link = simLinks[i];
          const src = typeof link.source === 'object' ? link.source : null;
          const tgt = typeof link.target === 'object' ? link.target : null;
          if (src && tgt) {
            const idx = i * 6;
            arr[idx] = src.x ?? 0;
            arr[idx + 1] = src.y ?? 0;
            arr[idx + 2] = src.z ?? 0;
            arr[idx + 3] = tgt.x ?? 0;
            arr[idx + 4] = tgt.y ?? 0;
            arr[idx + 5] = tgt.z ?? 0;
          }
        }
        posAttr.needsUpdate = true;
      }
    }

    // Update active edge lines — only edges where both endpoints are active
    if (activeLineGeo) {
      const posAttr = activeLineGeo.getAttribute('position') as THREE.BufferAttribute | undefined;
      if (posAttr) {
        const arr = posAttr.array as Float32Array;
        let activeCount = 0;
        for (let i = 0; i < simLinks.length; i++) {
          const link = simLinks[i];
          const src = typeof link.source === 'object' ? link.source : null;
          const tgt = typeof link.target === 'object' ? link.target : null;
          if (
            src &&
            tgt &&
            activeNodeIds.has(src.id) &&
            activeNodeIds.has(tgt.id) &&
            !(
              selectedNodeId !== null &&
              (selectedClusterIds.has(src.id) || selectedClusterIds.has(tgt.id))
            )
          ) {
            const idx = activeCount * 6;
            if (idx + 5 < arr.length) {
              arr[idx] = src.x ?? 0;
              arr[idx + 1] = src.y ?? 0;
              arr[idx + 2] = src.z ?? 0;
              arr[idx + 3] = tgt.x ?? 0;
              arr[idx + 4] = tgt.y ?? 0;
              arr[idx + 5] = tgt.z ?? 0;
              activeCount++;
            }
          }
        }
        posAttr.needsUpdate = true;
        activeLineGeo.setDrawRange(0, activeCount * 2);
      }
    }

    // Update selected edge lines — only edges connected to selected node
    if (selectedLineGeo) {
      const posAttr = selectedLineGeo.getAttribute('position') as THREE.BufferAttribute | undefined;
      if (posAttr) {
        const arr = posAttr.array as Float32Array;
        let selectedCount = 0;
        for (let i = 0; i < simLinks.length; i++) {
          const link = simLinks[i];
          const src = typeof link.source === 'object' ? link.source : null;
          const tgt = typeof link.target === 'object' ? link.target : null;
          if (src && tgt && selectedNodeId !== null && (src.id === selectedNodeId || tgt.id === selectedNodeId)) {
            const idx = selectedCount * 6;
            if (idx + 5 < arr.length) {
              arr[idx] = src.x ?? 0;
              arr[idx + 1] = src.y ?? 0;
              arr[idx + 2] = src.z ?? 0;
              arr[idx + 3] = tgt.x ?? 0;
              arr[idx + 4] = tgt.y ?? 0;
              arr[idx + 5] = tgt.z ?? 0;
              selectedCount++;
            }
          }
        }
        posAttr.needsUpdate = true;
        selectedLineGeo.setDrawRange(0, selectedCount * 2);
      }
    }

    // Render selected edges as cylinders so selected segments are visibly thicker.
    const selectedEdgeMesh = selectedEdgeMeshRef.current;
    if (selectedEdgeMesh) {
      const dummy = selectedEdgeDummyRef.current;
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      const dir = new THREE.Vector3();
      const mid = new THREE.Vector3();
      let selectedCount = 0;

      for (let i = 0; i < simLinks.length; i++) {
        const link = simLinks[i];
        const src = typeof link.source === 'object' ? link.source : null;
        const tgt = typeof link.target === 'object' ? link.target : null;
        if (!src || !tgt) continue;
        if (!(selectedNodeId !== null && (src.id === selectedNodeId || tgt.id === selectedNodeId))) continue;

        start.set(src.x ?? 0, src.y ?? 0, src.z ?? 0);
        end.set(tgt.x ?? 0, tgt.y ?? 0, tgt.z ?? 0);
        dir.subVectors(end, start);
        const length = dir.length();
        if (length < 1e-6) continue;

        mid.copy(start).add(end).multiplyScalar(0.5);
        dummy.position.copy(mid);
        dummy.quaternion.setFromUnitVectors(CYLINDER_BASE_DIR, dir.normalize());
        dummy.scale.set(1, length, 1);
        dummy.updateMatrix();

        selectedEdgeMesh.setMatrixAt(selectedCount, dummy.matrix);
        selectedCount++;
      }

      selectedEdgeMesh.count = selectedCount;
      selectedEdgeMesh.instanceMatrix.needsUpdate = true;
    }

    // Update tooltip position imperatively — no setState
    if (hoveredId !== null && tooltipRef.current) {
      const sn = simNodes.find((n) => n.id === hoveredId);
      if (sn) {
        tooltipRef.current.position.set(sn.x ?? 0, sn.y ?? 0, sn.z ?? 0);
      }
    }
  });

  return (
    <group>
      {/* Node meshes — each gets a ref by id, position driven by simulation in useFrame */}
      {nodes.map((node) => (
        <mesh
          key={node.id}
          position={node.position}
          ref={(m) => {
            if (m) meshRefs.current.set(node.id, m);
            else meshRefs.current.delete(node.id);
          }}
          onPointerOver={() => {
            setHoveredId(node.id);
            // Set initial tooltip text on hover start
            const sn = simNodesRef.current.find((n) => n.id === node.id);
            if (sn && tooltipLabelRef.current && tooltipCoordsRef.current) {
              tooltipLabelRef.current.text = sn.label;
              tooltipCoordsRef.current.text = `(${(sn.x ?? 0).toFixed(2)}, ${(sn.y ?? 0).toFixed(2)}, ${(sn.z ?? 0).toFixed(2)})`;
            }
          }}
          onPointerOut={() => setHoveredId(null)}
          onPointerDown={() => {
            const neighborIds = new Set<string>([node.id]);
            for (const edge of edges) {
              if (edge.sourceId === node.id) neighborIds.add(edge.targetId);
              if (edge.targetId === node.id) neighborIds.add(edge.sourceId);
            }

            const store = useAppStore.getState();
            store.setActiveNodes([...neighborIds]);
            store.setSelectedNode(node.id);
          }}
        >
          <sphereGeometry args={[0.15, 16, 16]} />
          <meshStandardMaterial color={node.color} emissive={node.color} emissiveIntensity={0.8} opacity={0.5} transparent />
        </mesh>
      ))}

      {/* Single LineSegments for all edges */}
      <lineSegments frustumCulled={false} renderOrder={2}>
        <bufferGeometry ref={lineGeoRef} />
        <lineBasicMaterial color="white" opacity={0.3} transparent depthTest={false} depthWrite={false} linewidth={1} />
      </lineSegments>

      {/* Active edge highlights */}
      <lineSegments frustumCulled={false} renderOrder={3}>
        <bufferGeometry ref={activeLineGeoRef} />
        <lineBasicMaterial color="cyan" opacity={1} depthTest={false} depthWrite={false} linewidth={1} />
      </lineSegments>

      {/* Selected cluster edge highlights */}
      <lineSegments frustumCulled={false} renderOrder={4}>
        <bufferGeometry ref={selectedLineGeoRef} />
        <lineBasicMaterial color="#dc2626" opacity={1} depthTest={false} depthWrite={false} linewidth={2} />
      </lineSegments>

      {/* Selected cluster thick edge overlays */}
      <instancedMesh ref={selectedEdgeMeshRef} args={[undefined, undefined, Math.max(edges.length, 1)]} frustumCulled={false} renderOrder={5}>
        <cylinderGeometry args={[SELECTED_EDGE_RADIUS, SELECTED_EDGE_RADIUS, 1, 8]} />
        <meshBasicMaterial color="#dc2626" opacity={1} transparent depthTest={false} depthWrite={false} />
      </instancedMesh>

      {/* Tooltip — visibility toggled, position updated imperatively in useFrame */}
      <group ref={tooltipRef} visible={hoveredId !== null}>
        <Text ref={tooltipLabelRef} position={[0, 0.35, 0]} fontSize={0.25} color="white" anchorY="bottom">
          {''}
        </Text>
        <Text ref={tooltipCoordsRef} position={[0, 0.12, 0]} fontSize={0.15} color="#aaaaaa" anchorY="bottom">
          {''}
        </Text>
      </group>
    </group>
  );
}

const AXIS_LENGTH = 8;
const TICK_SIZE = 0.15;

function AxisTicks({ axis }: { axis: 'x' | 'y' | 'z' }) {
  const ticks: [number, number, number][][] = [];
  for (let i = -AXIS_LENGTH; i <= AXIS_LENGTH; i++) {
    if (i === 0) continue;
    if (axis === 'x') {
      ticks.push([[i, -TICK_SIZE, 0], [i, TICK_SIZE, 0]]);
    } else if (axis === 'y') {
      ticks.push([[-TICK_SIZE, i, 0], [TICK_SIZE, i, 0]]);
    } else {
      ticks.push([[0, -TICK_SIZE, i], [0, TICK_SIZE, i]]);
    }
  }
  return (
    <>
      {ticks.map((pts, i) => (
        <Line key={i} points={pts} color="white" lineWidth={1} transparent opacity={0.05} />
      ))}
    </>
  );
}

function Axes() {
  return (
    <group>
      <Line points={[[-AXIS_LENGTH, 0, 0], [AXIS_LENGTH, 0, 0]]} color="white" lineWidth={1.5} transparent opacity={0.05} />
      <Line points={[[0, -AXIS_LENGTH, 0], [0, AXIS_LENGTH, 0]]} color="white" lineWidth={1.5} transparent opacity={0.05} />
      <Line points={[[0, 0, -AXIS_LENGTH], [0, 0, AXIS_LENGTH]]} color="white" lineWidth={1.5} transparent opacity={0.05} />
      <AxisTicks axis="x" />
      <AxisTicks axis="y" />
      <AxisTicks axis="z" />
    </group>
  );
}

const DEFAULT_CAM_POS = new THREE.Vector3(0, 2, 28);
const DEFAULT_TARGET = new THREE.Vector3(0, 0, 0);
const CAM_OFFSET_DISTANCE = 5;
const SELECTED_ZOOM_FACTOR = 0.55;
const MIN_SELECTED_DISTANCE = 1.8;

function CameraController({ simNodesRef, controlsRef }: {
  simNodesRef: React.RefObject<SimNode[]>;
  controlsRef: React.RefObject<any>;
}) {
  const { camera } = useThree();

  useFrame(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    if (!controls.target.equals(DEFAULT_TARGET)) {
      controls.target.copy(DEFAULT_TARGET);
      controls.update();
    }
  });

  useEffect(() => {
    let prevActiveIds: Set<string> = useAppStore.getState().activeNodeIds;
    let prevSelectedNodeId: string | null = useAppStore.getState().selectedNodeId;
    const unsub = useAppStore.subscribe((state) => {
      const activeNodeIds = state.activeNodeIds;
      const selectedNodeId = state.selectedNodeId;
      if (activeNodeIds === prevActiveIds && selectedNodeId === prevSelectedNodeId) return;
      const activeChanged = activeNodeIds !== prevActiveIds;
      const selectedChanged = selectedNodeId !== prevSelectedNodeId;
      prevActiveIds = activeNodeIds;
      prevSelectedNodeId = selectedNodeId;

      const controls = controlsRef.current;
      if (!controls) return;
      controls.target.copy(DEFAULT_TARGET);
      gsap.killTweensOf(camera.position);
      gsap.killTweensOf(controls.target);

      const simNodes = simNodesRef.current;

      if (selectedNodeId) {
        if (selectedChanged) {
          const selectedNode = simNodes.find((n) => n.id === selectedNodeId);
          if (selectedNode) {
            const selectedPos = new THREE.Vector3(selectedNode.x ?? 0, selectedNode.y ?? 0, selectedNode.z ?? 0);
            const currentCamPos = new THREE.Vector3(camera.position.x, camera.position.y, camera.position.z);
            const nextCamPos = currentCamPos.lerp(selectedPos, SELECTED_ZOOM_FACTOR);

            const targetVec = new THREE.Vector3(DEFAULT_TARGET.x, DEFAULT_TARGET.y, DEFAULT_TARGET.z);
            const toTarget = new THREE.Vector3().subVectors(nextCamPos, targetVec);
            const distanceToTarget = toTarget.length();
            if (distanceToTarget < MIN_SELECTED_DISTANCE && distanceToTarget > 0) {
              toTarget.setLength(MIN_SELECTED_DISTANCE);
              nextCamPos.copy(targetVec).add(toTarget);
            }

            gsap.to(camera.position, {
              x: nextCamPos.x,
              y: nextCamPos.y,
              z: nextCamPos.z,
              duration: 0.9,
              ease: 'power2.inOut',
              onUpdate: () => {
                controls.target.copy(DEFAULT_TARGET);
                controls.update();
              },
            });
          }
        }
        return;
      }

      if (!activeChanged) return;

      if (activeNodeIds.size === 0) {
          gsap.to(camera.position, {
            x: DEFAULT_CAM_POS.x,
            y: DEFAULT_CAM_POS.y,
            z: DEFAULT_CAM_POS.z,
            duration: 1.2,
            ease: 'power2.inOut',
            onUpdate: () => {
              controls.target.copy(DEFAULT_TARGET);
              controls.update();
            },
          });
          gsap.to(controls.target, {
            x: DEFAULT_TARGET.x,
            y: DEFAULT_TARGET.y,
            z: DEFAULT_TARGET.z,
            duration: 1.2,
            ease: 'power2.inOut',
            onUpdate: () => controls.update(),
          });
          return;
        }

        const activeNodes = simNodes.filter((n) => activeNodeIds.has(n.id));
        if (activeNodes.length === 0) return;

        // Compute centroid
        let cx = 0, cy = 0, cz = 0;
        for (const n of activeNodes) {
          cx += n.x ?? 0;
          cy += n.y ?? 0;
          cz += n.z ?? 0;
        }
        cx /= activeNodes.length;
        cy /= activeNodes.length;
        cz /= activeNodes.length;

        // Offset camera along direction from centroid to current camera position
        const dir = new THREE.Vector3().copy(camera.position).sub(new THREE.Vector3(cx, cy, cz));
        const len = dir.length();
        if (len > 0) dir.divideScalar(len);
        else dir.set(0, 0.3, 1).normalize();

        const targetCamPos = new THREE.Vector3(cx, cy, cz).add(dir.multiplyScalar(CAM_OFFSET_DISTANCE));

        gsap.to(camera.position, {
          x: targetCamPos.x,
          y: targetCamPos.y,
          z: targetCamPos.z,
          duration: 1.2,
          ease: 'power2.inOut',
          onUpdate: () => {
            controls.target.copy(DEFAULT_TARGET);
            controls.update();
          },
        });
    });
    return unsub;
  }, [camera, controlsRef, simNodesRef]);

  return null;
}

export default function Scene() {
  const nodes = useAppStore((state) => state.nodes);
  const edges = useAppStore((state) => state.edges);
  const controlsRef = useRef<any>(null);
  const simNodesRefForCamera = useRef<SimNode[]>([]);

  return (
    <Canvas camera={{ position: [0, 2, 32], fov: 50, near: 0.01, far: 200 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />

      <Axes />
      <ForceGraph nodes={nodes} edges={edges} simNodesRefOut={simNodesRefForCamera} />

      <mesh>
        <sphereGeometry args={[8, 32, 32]} />
        <meshBasicMaterial color="cyan" wireframe opacity={0.05} transparent />
      </mesh>

      <CameraController simNodesRef={simNodesRefForCamera} controlsRef={controlsRef} />
      <OrbitControls ref={controlsRef} makeDefault dampingFactor={0.1} target={[0, 0, 0]} enablePan zoomToCursor />
      <color attach="background" args={['#050510']} />
    </Canvas>
  );
}
