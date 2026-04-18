'use client'
import { useState, useRef, useCallback, useEffect } from 'react';
import Scene from '@/components/scene';
import { useAppStore, generateRandomHex, type GraphData } from '@/helpers/store';
import { FileUpload } from '@/components/file-upload';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

const MIN_SCREEN_WIDTH = 1200;

interface QuerySource {
  text: string;
  metadata: { chat_id: number; length: number; source: string };
}

type HeadHeatmap = {
  size: number;
  values: Float32Array;
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function toPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.floor(parsed);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function createHeadHeatmap(size: number): HeadHeatmap {
  return {
    size,
    values: new Float32Array(size * size),
  };
}

function ensureHeatmapGrid(grid: HeadHeatmap[][], layers: number, heads: number, seqLen: number): void {
  const safeLayers = Math.max(0, layers);
  const safeHeads = Math.max(0, heads);
  const safeSeqLen = Math.max(1, seqLen);

  while (grid.length < safeLayers) {
    grid.push([]);
  }

  for (let layerIndex = 0; layerIndex < safeLayers; layerIndex++) {
    const row = grid[layerIndex] ?? [];
    grid[layerIndex] = row;

    while (row.length < safeHeads) {
      row.push(createHeadHeatmap(safeSeqLen));
    }

    for (let headIndex = 0; headIndex < safeHeads; headIndex++) {
      const tile = row[headIndex];
      if (!tile || tile.size !== safeSeqLen) {
        row[headIndex] = createHeadHeatmap(safeSeqLen);
      }
    }
  }
}

function writeMatrix(heatmap: HeadHeatmap, matrixRows: unknown[]): void {
  const size = heatmap.size;

  for (let row = 0; row < Math.min(size, matrixRows.length); row++) {
    const rowValues = matrixRows[row];
    if (!Array.isArray(rowValues)) continue;

    const maxCol = Math.min(size, rowValues.length, row + 1);
    for (let col = 0; col < maxCol; col++) {
      heatmap.values[row * size + col] = clamp01(Number(rowValues[col]));
    }
  }
}

function writeRow(heatmap: HeadHeatmap, rowValues: unknown[], rowIndex: number): void {
  const size = heatmap.size;
  if (size <= 0) return;

  const safeRow = Math.max(0, Math.min(size - 1, rowIndex));
  const maxCol = Math.min(size, rowValues.length, safeRow + 1);
  for (let col = 0; col < maxCol; col++) {
    heatmap.values[safeRow * size + col] = clamp01(Number(rowValues[col]));
  }
}

function paintHeatmap(canvas: HTMLCanvasElement, heatmap: HeadHeatmap | null): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  if (!heatmap) {
    canvas.width = 1;
    canvas.height = 1;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 1, 1);
    return;
  }

  const { size, values } = heatmap;
  if (canvas.width !== size) canvas.width = size;
  if (canvas.height !== size) canvas.height = size;

  const image = ctx.createImageData(size, size);
  const data = image.data;

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const pixelIndex = (row * size + col) * 4;
      const value = col <= row ? clamp01(values[row * size + col]) : 0;
      const intensity = Math.round(value * 255);

      data[pixelIndex] = 0;
      data[pixelIndex + 1] = intensity;
      data[pixelIndex + 2] = intensity;
      data[pixelIndex + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function paintLayerAtlas(
  canvas: HTMLCanvasElement,
  layerHeatmaps: HeadHeatmap[] | undefined,
  heads: number,
  seqLen: number
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const safeHeads = Math.max(1, heads);
  const tileSize = Math.max(1, seqLen);
  const width = safeHeads * tileSize;
  const height = tileSize;

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  const image = ctx.createImageData(width, height);
  const data = image.data;

  for (let headIndex = 0; headIndex < safeHeads; headIndex++) {
    const heatmap = layerHeatmaps?.[headIndex] ?? null;
    const sourceSize = heatmap?.size ?? 0;

    for (let row = 0; row < tileSize; row++) {
      const sourceRow = row < sourceSize ? row : -1;
      for (let col = 0; col < tileSize; col++) {
        let value = 0;
        if (sourceRow >= 0 && col < sourceSize && col <= row && heatmap) {
          value = clamp01(heatmap.values[sourceRow * sourceSize + col]);
        }

        const intensity = Math.round(value * 255);
        const x = headIndex * tileSize + col;
        const y = row;
        const pixelIndex = (y * width + x) * 4;
        data[pixelIndex] = 0;
        data[pixelIndex + 1] = intensity;
        data[pixelIndex + 2] = intensity;
        data[pixelIndex + 3] = 255;
      }
    }
  }

  // Draw separators so head boundaries remain visible even in compressed view.
  for (let boundary = 1; boundary < safeHeads; boundary++) {
    const x = boundary * tileSize - 1;
    if (x < 0 || x >= width) continue;
    for (let y = 0; y < height; y++) {
      const pixelIndex = (y * width + x) * 4;
      data[pixelIndex] = 100;
      data[pixelIndex + 1] = 100;
      data[pixelIndex + 2] = 100;
      data[pixelIndex + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function AttentionOverview({
  heatmaps,
  layers,
  heads,
  seqLen,
  version,
  onPick,
  selectedLayerIndex,
}: {
  heatmaps: HeadHeatmap[][];
  layers: number;
  heads: number;
  seqLen: number;
  version: number;
  onPick?: (layerIndex: number) => void;
  selectedLayerIndex?: number;
}) {
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([]);
  const [hoverLabel, setHoverLabel] = useState('Hover a layer plane to inspect');
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStateRef = useRef({
    active: false,
    pointerId: -1,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
    moved: false,
  });

  const safeLayers = Math.max(1, layers);
  const safeHeads = Math.max(1, heads);
  const safeSeqLen = Math.max(1, seqLen);

  useEffect(() => {
    for (let layerIndex = 0; layerIndex < safeLayers; layerIndex++) {
      const canvas = canvasRefs.current[layerIndex];
      if (!canvas) continue;
      paintLayerAtlas(canvas, heatmaps[layerIndex], safeHeads, safeSeqLen);
    }
  }, [heatmaps, safeLayers, safeHeads, safeSeqLen, version]);

  useEffect(() => {
    setPan({ x: 0, y: 0 });
    setIsDragging(false);
    dragStateRef.current.active = false;
    dragStateRef.current.moved = false;
  }, [safeLayers, safeHeads]);

  const depthStep = safeLayers > 1 ? Math.max(24, Math.min(124, Math.floor(1450 / safeLayers))) : 0;
  const yStep = safeLayers > 1 ? Math.max(9, Math.min(34, Math.floor(560 / safeLayers))) : 0;
  const stageScale = Math.max(0.22, Math.min(0.9, 1.08 - safeLayers * 0.026));

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    dragStateRef.current = {
      active: true,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: pan.x,
      originY: pan.y,
      moved: false,
    };
    setIsDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (!dragState.active || dragState.pointerId !== event.pointerId) return;

    const dx = event.clientX - dragState.startX;
    const dy = event.clientY - dragState.startY;
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
      dragStateRef.current.moved = true;
    }
    setPan({ x: dragState.originX + dx, y: dragState.originY + dy });
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (dragState.pointerId === event.pointerId) {
      if (!dragState.moved && onPick) {
        const target = document.elementFromPoint(event.clientX, event.clientY) as HTMLElement | null;
        const layerEl = target?.closest('[data-layer-index]') as HTMLElement | null;
        if (layerEl) {
          const layerRaw = Number(layerEl.dataset.layerIndex);
          const layerIndex = Math.max(0, Math.min(safeLayers - 1, Number.isFinite(layerRaw) ? layerRaw : 0));
          onPick(layerIndex);
        }
      }
      dragStateRef.current.active = false;
      setIsDragging(false);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    }
  };

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div
        className={`absolute inset-0 flex items-center justify-center ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
        style={{ perspective: '1500px' }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onDoubleClick={() => setPan({ x: 0, y: 0 })}
      >
        <div
          className="relative h-[128%] w-[132%]"
          style={{
            transformStyle: 'preserve-3d',
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${stageScale}) rotateX(57deg) rotateZ(-10deg)`,
          }}
        >
          {Array.from({ length: safeLayers }).map((_, layerIndex) => (
            <div
              key={`layer-plane-${layerIndex}`}
              data-layer-index={layerIndex}
              className={`absolute left-1/2 top-1/2 overflow-hidden rounded-sm bg-black/85 shadow-[0_0_12px_rgba(34,211,238,0.08)] ${
                selectedLayerIndex === layerIndex ? 'border border-cyan-300/60' : 'border border-cyan-300/20'
              }`}
              style={{
                width: '96%',
                height: '78%',
                transform: `translate(-50%, -50%) translateY(${layerIndex * yStep}px) translateZ(${(safeLayers - 1 - layerIndex) * depthStep}px)`,
              }}
            >
              <canvas
                ref={(node) => {
                  canvasRefs.current[layerIndex] = node;
                }}
                className="h-full w-full [image-rendering:pixelated]"
                onMouseMove={(event) => {
                  const width = event.currentTarget.clientWidth || 1;
                  const rawHead = Math.floor((event.nativeEvent.offsetX / width) * safeHeads);
                  const headIndex = Math.max(0, Math.min(safeHeads - 1, rawHead));
                  setHoverLabel(`Layer ${layerIndex + 1} • Head ${headIndex + 1}`);
                }}
                onMouseLeave={() => setHoverLabel('Hover a layer plane to inspect')}
              />
              <div className="pointer-events-none absolute left-1 top-1 rounded bg-black/70 px-1 py-0.5 text-[10px] text-cyan-200">
                L{layerIndex + 1}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="pointer-events-none absolute left-2 top-2 text-[10px] text-muted-foreground">
        3D stack: depth = layers, x-axis = heads, drag to pan (double-click to reset)
      </div>
      <div className="pointer-events-none absolute right-2 top-2 rounded bg-black/65 px-2 py-1 text-[11px] text-cyan-200">
        {hoverLabel}
      </div>
    </div>
  );
}

function SingleHeadView({
  heatmap,
  layerIndex,
  headIndex,
  version,
}: {
  heatmap: HeadHeatmap | null;
  layerIndex: number;
  headIndex: number;
  version: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    paintHeatmap(canvasRef.current, heatmap);
  }, [heatmap, version]);

  const size = heatmap?.size ?? 1;

  return (
    <div className="relative flex h-full w-full items-center justify-center overflow-hidden p-2">
      <div className="relative h-full w-full max-h-[88vh] max-w-[88vh]">
        <canvas
          ref={canvasRef}
          width={size}
          height={size}
          className="h-full w-full rounded border border-cyan-300/40 bg-black [image-rendering:pixelated]"
        />
        <div className="pointer-events-none absolute left-2 top-2 rounded bg-black/65 px-2 py-1 text-[11px] text-cyan-200">
          Layer {layerIndex + 1} • Head {headIndex + 1}
        </div>
      </div>
    </div>
  );
}

function LayerOverview({
  layerHeatmaps,
  heads,
  version,
  selectedHeadIndex,
  onPickHead,
}: {
  layerHeatmaps: HeadHeatmap[] | undefined;
  heads: number;
  version: number;
  selectedHeadIndex: number;
  onPickHead?: (headIndex: number) => void;
}) {
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([]);
  const safeHeads = Math.max(1, heads);

  useEffect(() => {
    for (let headIndex = 0; headIndex < safeHeads; headIndex++) {
      const canvas = canvasRefs.current[headIndex];
      if (!canvas) continue;
      paintHeatmap(canvas, layerHeatmaps?.[headIndex] ?? null);
    }
  }, [layerHeatmaps, safeHeads, version]);

  const columns =
    safeHeads >= 28 ? 8 :
    safeHeads >= 16 ? 6 :
    safeHeads >= 9 ? 5 :
    safeHeads >= 6 ? 4 :
    safeHeads;

  return (
    <div className="h-full w-full overflow-hidden p-2">
      <div
        className="grid h-full w-full gap-2"
        style={{
          gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
          gridAutoRows: '1fr',
        }}
      >
        {Array.from({ length: safeHeads }).map((_, headIndex) => (
          <button
            key={`layer-head-${headIndex}`}
            type="button"
            onClick={() => onPickHead?.(headIndex)}
            className={`group relative overflow-hidden rounded border bg-black/80 ${
              selectedHeadIndex === headIndex
                ? 'border-cyan-300/70 shadow-[0_0_10px_rgba(34,211,238,0.28)]'
                : 'border-cyan-300/25'
            }`}
          >
            <canvas
              ref={(node) => {
                canvasRefs.current[headIndex] = node;
              }}
              className="h-full w-full [image-rendering:pixelated]"
            />
            <div className="pointer-events-none absolute left-1 top-1 rounded bg-black/70 px-1 py-0.5 text-[10px] text-cyan-200">
              H{headIndex + 1}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [queryText, setQueryText] = useState('');
  const [aiQuery, setAiQuery] = useState('');
  const [aiK, setAiK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(200);
  const [isAttentionPopoverOpen, setIsAttentionPopoverOpen] = useState(false);
  const [attentionViewMode, setAttentionViewMode] = useState<'stack' | 'layer' | 'single'>('stack');
  const [selectedLayerIndex, setSelectedLayerIndex] = useState(0);
  const [selectedHeadIndex, setSelectedHeadIndex] = useState(0);
  const [showAttentionMeta, setShowAttentionMeta] = useState(false);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [aiSources, setAiSources] = useState<QuerySource[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [attentionStatus, setAttentionStatus] = useState('Idle');
  const [promptTokens, setPromptTokens] = useState<string[]>([]);
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);
  const [seqLen, setSeqLen] = useState(0);
  const [totalLayers, setTotalLayers] = useState(0);
  const [totalHeads, setTotalHeads] = useState(0);
  const [heatmapVersion, setHeatmapVersion] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [hasUploadedPdf, setHasUploadedPdf] = useState(false);
  const [pendingUploadFile, setPendingUploadFile] = useState<File | null>(null);
  const [isScreenWideEnough, setIsScreenWideEnough] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);
  const heatmapsRef = useRef<HeadHeatmap[][]>([]);
  const seqLenRef = useRef(0);
  const totalLayersRef = useRef(0);
  const totalHeadsRef = useRef(0);
  const { nodes, edges, isLoading, setLoading, setActiveNodes, setAiSourceNodes, selectedNodeId, setSelectedNode, loadGraph } = useAppStore();

  const pinSourceNodes = useCallback((sources: QuerySource[]) => {
    const { nodes: currentNodes, edges: currentEdges } = useAppStore.getState();
    const sourceNodeIds = new Set<string>();
    for (const src of sources) {
      for (const node of currentNodes) {
        if (node.text === src.text) {
          sourceNodeIds.add(node.id);
          break;
        }
      }
    }

    const matchedIds = new Set<string>(sourceNodeIds);
    for (const edge of currentEdges) {
      if (matchedIds.has(edge.sourceId)) matchedIds.add(edge.targetId);
      if (matchedIds.has(edge.targetId)) matchedIds.add(edge.sourceId);
    }
    useAppStore.getState().setAiSourceNodes([...matchedIds]);
    useAppStore.getState().setActiveNodes([...matchedIds]);
  }, []);

  const resetAttentionState = useCallback(() => {
    seqLenRef.current = 0;
    totalLayersRef.current = 0;
    totalHeadsRef.current = 0;
    heatmapsRef.current = [];
    setAttentionViewMode('stack');
    setSelectedLayerIndex(0);
    setSelectedHeadIndex(0);
    setPromptTokens([]);
    setGeneratedTokens([]);
    setSeqLen(0);
    setTotalLayers(0);
    setTotalHeads(0);
    setHeatmapVersion((v) => v + 1);
  }, []);

  const handleAiQuery = useCallback(() => {
    if (!aiQuery.trim() || isQuerying) return;

    if (!process.env.NEXT_PUBLIC_HOSTNAME) {
      setAiAnswer('Missing NEXT_PUBLIC_HOSTNAME.');
      return;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsQuerying(true);
    setIsAttentionPopoverOpen(true);
    setAttentionStatus('Connecting to /ws/attention...');
    setAiAnswer(null);
    setAiSources([]);
    resetAttentionState();

    const ws = new WebSocket(`wss://${process.env.NEXT_PUBLIC_HOSTNAME}/ws/attention`);
    wsRef.current = ws;
    let completed = false;

    ws.onopen = () => {
      setAttentionStatus('Connected. Streaming attention heads...');
      ws.send(JSON.stringify({ query: aiQuery.trim(), k: aiK, max_tokens: maxTokens }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as unknown;
        if (!isRecord(msg)) return;

        const eventName = typeof msg.event === 'string' ? msg.event : '';
        const data = msg.data;

        if (eventName === 'graph' && isRecord(data)) {
          const rawNodes = Array.isArray(data.nodes) ? data.nodes : [];
          const rawEdges = Array.isArray(data.edges) ? data.edges : [];
          const rawPath = Array.isArray(data.path) ? data.path : [];

          const graphNodes = rawNodes
            .filter((n): n is Record<string, unknown> => isRecord(n))
            .map((n) => {
              const x = Number(n.x);
              const y = Number(n.y);
              const z = Number(n.z);
              const id = typeof n.id === 'string' ? n.id : `node-${Math.random().toString(16).slice(2)}`;
              const label = typeof n.label === 'string' ? n.label : id;
              const text = typeof n.text === 'string' ? n.text : '';
              return {
                id,
                label,
                text,
                position: [
                  Number.isFinite(x) ? x : 0,
                  Number.isFinite(y) ? y : 0,
                  Number.isFinite(z) ? z : 0,
                ] as [number, number, number],
                color: n.is_source === true ? '#22d3ee' : generateRandomHex(),
              };
            });

          const graphEdges = rawEdges
            .filter((e): e is Record<string, unknown> => isRecord(e))
            .map((e) => ({
              sourceId: typeof e.source === 'string' ? e.source : '',
              targetId: typeof e.target === 'string' ? e.target : '',
            }))
            .filter((e) => e.sourceId !== '' && e.targetId !== '');

          loadGraph(graphNodes, graphEdges);

          const pathIds = rawPath.filter((id): id is string => typeof id === 'string');
          if (pathIds.length > 0) {
            setAiSourceNodes(pathIds);
            setActiveNodes(pathIds);
          }

          setAttentionStatus(`Graph received: ${graphNodes.length} nodes, ${graphEdges.length} edges.`);
          return;
        }

        if (eventName === 'tokens' && isRecord(data)) {
          const incomingTokens = Array.isArray(data.prompt_tokens)
            ? data.prompt_tokens.map((token) => String(token))
            : Array.isArray(data.tokens)
              ? data.tokens.map((token) => String(token))
              : [];
          const nextSeqLen = Math.max(1, toPositiveInt(data.seq_len, incomingTokens.length));
          const layers = toPositiveInt(data.total_layers, toPositiveInt(data.num_layers, 0));
          const heads = toPositiveInt(data.total_heads, toPositiveInt(data.num_heads, 0));

          seqLenRef.current = nextSeqLen;
          totalLayersRef.current = layers;
          totalHeadsRef.current = heads;

          ensureHeatmapGrid(
            heatmapsRef.current,
            layers,
            heads,
            nextSeqLen
          );

          setPromptTokens(incomingTokens.slice(0, nextSeqLen));
          setGeneratedTokens([]);
          setSeqLen(nextSeqLen);
          setTotalLayers(layers);
          setTotalHeads(heads);
          setAttentionStatus(`Streaming attention for ${incomingTokens.length} prompt tokens...`);
          setHeatmapVersion((v) => v + 1);
          return;
        }

        if (eventName === 'attention' && isRecord(data)) {
          const step = Math.max(0, Math.floor(Number(data.step) || 0));
          const token = typeof data.token === 'string' ? data.token : '';
          if (token) {
            setGeneratedTokens((prev) => [...prev, token]);
          }

          const incomingTokens = Array.isArray(data.tokens)
            ? data.tokens.map((token) => String(token))
            : [];
          const declaredSeqLen = toPositiveInt(data.seq_len, incomingTokens.length);
          const nextSeqLen = Math.max(1, declaredSeqLen || incomingTokens.length);

          const gridRaw = Array.isArray(data.attention_grid) ? data.attention_grid : [];
          const layers = toPositiveInt(data.num_layers, toPositiveInt(data.total_layers, gridRaw.length));
          const headsFromGrid =
            Array.isArray(gridRaw[0]) ? (gridRaw[0] as unknown[]).length : 0;
          const heads = toPositiveInt(data.num_heads, toPositiveInt(data.total_heads, headsFromGrid));

          seqLenRef.current = nextSeqLen;
          totalLayersRef.current = layers;
          totalHeadsRef.current = heads;

          ensureHeatmapGrid(
            heatmapsRef.current,
            layers,
            heads,
            nextSeqLen
          );

          for (let layerIndex = 0; layerIndex < Math.min(layers, gridRaw.length); layerIndex++) {
            const layerHeads = gridRaw[layerIndex];
            if (!Array.isArray(layerHeads)) continue;

            for (let headIndex = 0; headIndex < Math.min(heads, layerHeads.length); headIndex++) {
              const matrixRows = layerHeads[headIndex];
              const heatmap = heatmapsRef.current[layerIndex]?.[headIndex];
              if (!heatmap || !Array.isArray(matrixRows)) continue;
              writeMatrix(heatmap, matrixRows as unknown[]);
            }
          }

          setPromptTokens(incomingTokens.slice(0, nextSeqLen));
          setSeqLen(nextSeqLen);
          setTotalLayers(layers);
          setTotalHeads(heads);
          setHeatmapVersion((v) => v + 1);
          setAttentionStatus(`Streaming attention step ${step + 1}...`);
          return;
        }

        if (eventName === 'gen_step' && isRecord(data)) {
          const step = Math.max(0, Math.floor(Number(data.step) || 0));
          const token = typeof data.token === 'string' ? data.token : '';
          if (token) {
            setGeneratedTokens((prev) => [...prev, token]);
          }

          const layerPayloads = Array.isArray(data.layers) ? data.layers : [];
          let nextMaxHeadCount = totalHeadsRef.current;

          for (const layerPayload of layerPayloads) {
            if (!isRecord(layerPayload)) continue;

            const layerIndex = Math.max(0, Math.floor(Number(layerPayload.layer) || 0));
            const headWeightsRaw = Array.isArray(layerPayload.head_weights)
              ? layerPayload.head_weights
              : [];

            if (layerIndex + 1 > totalLayersRef.current) {
              totalLayersRef.current = layerIndex + 1;
            }

            if (headWeightsRaw.length > nextMaxHeadCount) {
              nextMaxHeadCount = headWeightsRaw.length;
            }

            ensureHeatmapGrid(
              heatmapsRef.current,
              totalLayersRef.current,
              nextMaxHeadCount,
              Math.max(1, seqLenRef.current)
            );

            for (let headIndex = 0; headIndex < headWeightsRaw.length; headIndex++) {
              const headData = headWeightsRaw[headIndex];
              const heatmap = heatmapsRef.current[layerIndex]?.[headIndex];
              if (!heatmap || !Array.isArray(headData)) continue;

              if (headData.length > 0 && Array.isArray(headData[0])) {
                writeMatrix(heatmap, headData as unknown[]);
              } else {
                writeRow(heatmap, headData as unknown[], step);
              }
            }
          }

          totalHeadsRef.current = nextMaxHeadCount;
          setTotalLayers(totalLayersRef.current);
          setTotalHeads(nextMaxHeadCount);
          setAttentionStatus(`Streaming step ${step + 1}...`);
          setHeatmapVersion((v) => v + 1);
          return;
        }

        if (eventName === 'answer' && isRecord(data)) {
          const answer = typeof data.answer === 'string' ? data.answer : '';
          const sources: QuerySource[] = Array.isArray(data.sources)
            ? (data.sources as QuerySource[])
            : [];
          setAiAnswer(answer);
          setAiSources(sources);
          pinSourceNodes(sources);
          setAttentionStatus('Answer received.');
          completed = true;
          setIsQuerying(false);
          setIsAttentionPopoverOpen(false);
          resetAttentionState();
          ws.close();
          return;
        }

        if (eventName === 'attention_done' && isRecord(data)) {
          const layers = toPositiveInt(
            data.total_layers,
            toPositiveInt(data.num_layers, totalLayersRef.current || 0)
          );
          const heads = toPositiveInt(
            data.total_heads,
            toPositiveInt(data.num_heads, totalHeadsRef.current || 0)
          );

          totalLayersRef.current = layers;
          totalHeadsRef.current = heads;

          if (layers > 0 && heads > 0) {
            ensureHeatmapGrid(
              heatmapsRef.current,
              layers,
              heads,
              Math.max(1, seqLenRef.current)
            );
          }

          setTotalLayers(layers);
          setTotalHeads(heads);
          setHeatmapVersion((v) => v + 1);
          setAttentionStatus(`Attention complete: ${layers} layers x ${heads} heads.`);
          completed = true;
          setIsQuerying(false);
          setIsAttentionPopoverOpen(false);
          resetAttentionState();
          ws.close();
        }
      } catch {
        setAttentionStatus('Received malformed websocket payload.');
      }
    };

    ws.onerror = () => {
      setAiAnswer('Failed to connect to attention service.');
      setAttentionStatus('WebSocket error while streaming attention.');
      setIsQuerying(false);
      setIsAttentionPopoverOpen(false);
      resetAttentionState();
    };

    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
      if (!completed) {
        setIsQuerying(false);
        setIsAttentionPopoverOpen(false);
      }
    };
  }, [
    aiK,
    aiQuery,
    isQuerying,
    loadGraph,
    maxTokens,
    pinSourceNodes,
    resetAttentionState,
    setActiveNodes,
    setAiSourceNodes,
  ]);

  const API_BASE = `https://${process.env.NEXT_PUBLIC_HOSTNAME}`;

  const handleLoadGraph = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/graph`);
      const data: GraphData = await res.json();
      const graphNodes = data.nodes.map((n) => ({
        id: n.id,
        label: n.label,
        text: n.text || '',
        position: [n.x, n.y, n.z] as [number, number, number],
        color: generateRandomHex(),
      }));
      const graphEdges = data.edges.map((e) => ({
        sourceId: e.source,
        targetId: e.target,
      }));
      loadGraph(graphNodes, graphEdges);
    } catch (err) {
      console.error('Failed to load graph:', err);
    } finally {
      setLoading(false);
    }
  }, [loadGraph, setLoading]);

  const handleUploadFileSelection = useCallback((files: File[]) => {
    const file = files?.[0];
    if (!file || isUploading) return;
    setHasUploadedPdf(false);
    setPendingUploadFile(file);
    setUploadStatus(`Ready to upload: ${file.name}`);
  }, [isUploading]);

  const handleConfirmUpload = useCallback(() => {
    if (!pendingUploadFile || isUploading) return;
    setHasUploadedPdf(false);
    setIsUploading(true);
    setUploadStatus('Connecting to upload service...');

    const uploadWs = new WebSocket(`wss://${process.env.NEXT_PUBLIC_HOSTNAME}/ws/upload`);

    uploadWs.onopen = () => {
      setUploadStatus('Connected. Uploading PDF bytes...');
      uploadWs.send(pendingUploadFile);
    };

    uploadWs.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        if (typeof parsed?.data === 'string') setUploadStatus(parsed.data);
        if (parsed?.event === 'done') {
          setHasUploadedPdf(true);
          setPendingUploadFile(null);
          setIsUploading(false);
          void handleLoadGraph();
          uploadWs.close();
        }
      } catch {
        if (typeof msg.data === 'string') setUploadStatus(msg.data);
      }
    };

    uploadWs.onerror = () => {
      setUploadStatus('Upload failed. Check that the server is running.');
      setHasUploadedPdf(false);
      setIsUploading(false);
    };

    uploadWs.onclose = () => {
      setIsUploading(false);
    };
  }, [handleLoadGraph, isUploading, pendingUploadFile]);

  useEffect(() => {
    const checkViewportWidth = () => {
      setIsScreenWideEnough(window.innerWidth >= MIN_SCREEN_WIDTH);
    };
    checkViewportWidth();
    window.addEventListener('resize', checkViewportWidth);
    return () => window.removeEventListener('resize', checkViewportWidth);
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    setSelectedLayerIndex((prev) => Math.max(0, Math.min(prev, Math.max(totalLayers - 1, 0))));
  }, [totalLayers]);

  useEffect(() => {
    setSelectedHeadIndex((prev) => Math.max(0, Math.min(prev, Math.max(totalHeads - 1, 0))));
  }, [totalHeads]);

  const maxLayerIndex = Math.max(totalLayers - 1, 0);
  const maxHeadIndex = Math.max(totalHeads - 1, 0);
  const activeLayerIndex = Math.max(0, Math.min(selectedLayerIndex, maxLayerIndex));
  const activeHeadIndex = Math.max(0, Math.min(selectedHeadIndex, maxHeadIndex));

  if (!isScreenWideEnough) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#050510] p-6 text-center text-white">
        <div>
          <h1 className="mb-3 text-2xl font-bold">Wider Screen Required</h1>
          <p className="text-base opacity-80">
            This experience requires a minimum width of {MIN_SCREEN_WIDTH}px.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      {/* LAYER 1: 3D Canvas */}
      <div className="absolute inset-0 z-0">
        <Scene />
      </div>

      {/* LAYER 2: HTML UI Overlay */}
      <div className="pointer-events-none absolute inset-0 z-10 flex flex-col justify-between p-6">
        {/* Top Header */}
        <div className="pointer-events-auto w-[min(360px,100%)]">
          <Card className="bg-black/60 backdrop-blur-xl border-white/10">
            <CardHeader className="pb-2">
              <CardTitle className="text-2xl font-bold text-white">AI Data Visualizer</CardTitle>
              <p className="text-sm text-muted-foreground">Rotate to explore the latent space.</p>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-xs text-muted-foreground">Nodes in scene: {nodes.length}</p>
              <p className="mt-1 text-xs text-muted-foreground">Attention: {attentionStatus}</p>
            </CardContent>
          </Card>
        </div>

        {/* Bottom Controls */}
        <div className="flex items-end justify-between gap-4 flex-wrap">
          {/* Upload Panel */}
          <Card className="pointer-events-auto w-[min(360px,100%)] bg-black/60 backdrop-blur-xl border-white/10">
            <CardContent className="p-3">
              <FileUpload className="p-4" onChange={handleUploadFileSelection} />
              <p className="mt-2 min-h-4.5 px-1 text-xs text-muted-foreground">
                {isUploading ? `Uploading: ${uploadStatus}` : uploadStatus}
              </p>
              <Button
                className="mt-2 w-full"
                variant={isUploading || !pendingUploadFile ? 'secondary' : 'default'}
                disabled={isUploading || !pendingUploadFile}
                onClick={handleConfirmUpload}
              >
                {isUploading ? 'Uploading...' : 'Confirm Upload'}
              </Button>
            </CardContent>
          </Card>

          {/* Center Controls */}
          <div className="flex flex-wrap items-end justify-center gap-3">
            {/* Load Graph */}
            <Card className="pointer-events-auto bg-black/60 backdrop-blur-xl border-white/10">
              <CardContent className="p-3">
                <Button
                  variant={isLoading || isUploading || !hasUploadedPdf ? 'secondary' : 'default'}
                  disabled={isLoading || isUploading || !hasUploadedPdf}
                  onClick={handleLoadGraph}
                  className="bg-cyan-500 text-black hover:bg-cyan-400 disabled:bg-secondary disabled:text-muted-foreground"
                >
                  {isLoading ? 'Loading...' : 'Load Graph'}
                </Button>
              </CardContent>
            </Card>

            {/* Search */}
            <Card className="pointer-events-auto bg-black/60 backdrop-blur-xl border-white/10">
              <CardContent className="p-3">
                <form
                  className="flex gap-2"
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (!queryText) return;
                    const { nodes: currentNodes, edges: currentEdges } = useAppStore.getState();
                    const query = queryText.toLowerCase();
                    const matched = currentNodes.filter((n) =>
                      n.label.toLowerCase().includes(query) || n.text.toLowerCase().includes(query)
                    );
                    if (matched.length === 0) {
                      setActiveNodes([]);
                      return;
                    }
                    const matchedIds = new Set(matched.map((n) => n.id));
                    for (const edge of currentEdges) {
                      if (matchedIds.has(edge.sourceId)) matchedIds.add(edge.targetId);
                      if (matchedIds.has(edge.targetId)) matchedIds.add(edge.sourceId);
                    }
                    setActiveNodes([...matchedIds]);
                  }}
                >
                  <Input
                    placeholder="Search nodes..."
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    className="w-50 bg-black/50 border-white/10 text-white placeholder:text-white/40"
                  />
                  <Button
                    type="submit"
                    disabled={nodes.length === 0}
                    className="bg-orange-500 text-white hover:bg-orange-400 disabled:bg-secondary disabled:text-muted-foreground"
                  >
                    Search
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* AI Query */}
            <Card className="pointer-events-auto bg-black/60 backdrop-blur-xl border-white/10">
              <CardContent className="p-3">
                <form
                  className="flex items-start gap-2"
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleAiQuery();
                  }}
                >
                  <div className="flex flex-col gap-2">
                    <Input
                      placeholder="Ask a question..."
                      value={aiQuery}
                      onChange={(e) => setAiQuery(e.target.value)}
                      className="w-72 bg-black/50 border-white/10 text-white placeholder:text-white/40"
                    />
                    <div className="flex items-center gap-3">
                      <span className="shrink-0 text-xs text-muted-foreground">k: {aiK}</span>
                      <Slider
                        min={5}
                        max={20}
                        step={1}
                        value={[aiK]}
                        className="w-32"
                        onValueChange={(val) => {
                          if (typeof val === 'number') setAiK(val);
                          else if (Array.isArray(val) && typeof val[0] === 'number') setAiK(val[0]);
                        }}
                      />
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-muted-foreground">max:</span>
                        <Input
                          type="number"
                          min={1}
                          max={2048}
                          value={maxTokens}
                          onChange={(e) => setMaxTokens(toPositiveInt(e.target.value, 200))}
                          className="h-7 w-24 bg-black/50 border-white/10 text-white"
                        />
                      </div>
                    </div>
                  </div>
                  <Button
                    type="submit"
                    disabled={isQuerying || !aiQuery.trim()}
                    className="self-start bg-purple-500 text-white hover:bg-purple-400 disabled:bg-secondary disabled:text-muted-foreground"
                  >
                    {isQuerying ? 'Thinking...' : 'Ask AI'}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* LAYER 3: Attention Heatmap Popover */}
      {isAttentionPopoverOpen && (
        <div className="pointer-events-auto fixed inset-0 z-40 bg-black/75 p-3 backdrop-blur-sm">
          <Card className="flex h-full w-full flex-col overflow-hidden border-cyan-300/20 bg-black/85">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-sm font-medium uppercase tracking-widest text-muted-foreground">
                  Attention Heads
                </CardTitle>
                <Button
                  type="button"
                  size="xs"
                  variant="secondary"
                  onClick={() => {
                    if (wsRef.current) {
                      wsRef.current.close();
                      wsRef.current = null;
                    }
                    setIsQuerying(false);
                    setIsAttentionPopoverOpen(false);
                    setAttentionStatus('Idle');
                    resetAttentionState();
                  }}
                >
                  Close
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">{attentionStatus}</p>
              <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
                <span>seq_len: {seqLen || '-'}</span>
                <span>layers: {totalLayers || '-'}</span>
                <span>heads: {totalHeads || '-'}</span>
              </div>
            </CardHeader>
            <CardContent className="flex min-h-0 flex-1 flex-col gap-2">
              {totalLayers > 0 && totalHeads > 0 && (
                <div className="flex flex-wrap items-center gap-2 rounded border border-white/10 bg-black/40 p-2">
                  <span className="text-[11px] text-muted-foreground">View:</span>
                  <Button
                    type="button"
                    size="xs"
                    variant={attentionViewMode === 'stack' ? 'default' : 'secondary'}
                    onClick={() => setAttentionViewMode('stack')}
                  >
                    3D Stack
                  </Button>
                  <Button
                    type="button"
                    size="xs"
                    variant={attentionViewMode === 'layer' ? 'default' : 'secondary'}
                    onClick={() => setAttentionViewMode('layer')}
                    disabled={totalLayers <= 0}
                  >
                    Layer Only
                  </Button>
                  <Button
                    type="button"
                    size="xs"
                    variant={attentionViewMode === 'single' ? 'default' : 'secondary'}
                    onClick={() => setAttentionViewMode('single')}
                    disabled={totalLayers <= 0 || totalHeads <= 0}
                  >
                    Single Head
                  </Button>

                  <span className="ml-2 text-[11px] text-muted-foreground">Layer</span>
                  <Input
                    type="number"
                    min={1}
                    max={Math.max(totalLayers, 1)}
                    value={activeLayerIndex + 1}
                    onChange={(e) => setSelectedLayerIndex(Math.max(0, toPositiveInt(e.target.value, 1) - 1))}
                    className="h-7 w-20 bg-black/60 text-xs"
                  />

                  <span className="text-[11px] text-muted-foreground">Head</span>
                  <Input
                    type="number"
                    min={1}
                    max={Math.max(totalHeads, 1)}
                    value={activeHeadIndex + 1}
                    onChange={(e) => setSelectedHeadIndex(Math.max(0, toPositiveInt(e.target.value, 1) - 1))}
                    className="h-7 w-20 bg-black/60 text-xs"
                  />

                  <Button
                    type="button"
                    size="xs"
                    variant="secondary"
                    onClick={() => setShowAttentionMeta((prev) => !prev)}
                  >
                    {showAttentionMeta ? 'Hide token strips' : 'Show token strips'}
                  </Button>
                </div>
              )}

              {showAttentionMeta && (
                <div className="grid gap-2 xl:grid-cols-2">
                  <div>
                    <p className="mb-1 text-[11px] text-muted-foreground">Sequence Window ({promptTokens.length})</p>
                    <div className="h-12 overflow-y-auto rounded border border-white/10 bg-black/40 p-2">
                      <div className="flex flex-wrap gap-1.5">
                        {promptTokens.length === 0 ? (
                          <p className="text-xs text-muted-foreground">No sequence tokens yet.</p>
                        ) : (
                          promptTokens.map((token, idx) => (
                            <span
                              key={`${token}-${idx}`}
                              className="rounded border border-cyan-400/25 bg-cyan-500/10 px-1.5 py-0.5 font-mono text-[11px]"
                            >
                              {token}
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  </div>
                  <div>
                    <p className="mb-1 text-[11px] text-muted-foreground">Generated Tokens ({generatedTokens.length})</p>
                    <div className="h-12 overflow-y-auto rounded border border-white/10 bg-black/40 p-2">
                      {generatedTokens.length === 0 ? (
                        <p className="text-xs text-muted-foreground">No generated tokens yet.</p>
                      ) : (
                        <p className="font-mono text-xs leading-relaxed text-white/85">{generatedTokens.join('')}</p>
                      )}
                    </div>
                  </div>
                </div>
              )}

              <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                <span>0.0</span>
                <div className="h-2 w-48 rounded-full bg-linear-to-r from-black to-cyan-400" />
                <span>1.0</span>
              </div>

              {totalLayers <= 0 || totalHeads <= 0 ? (
                <p className="text-xs text-muted-foreground">Waiting for head-weight data...</p>
              ) : (
                <div className="min-h-0 flex-1 overflow-hidden rounded border border-white/10 bg-black/55">
                  {attentionViewMode === 'single' ? (
                    <SingleHeadView
                      heatmap={heatmapsRef.current[activeLayerIndex]?.[activeHeadIndex] ?? null}
                      layerIndex={activeLayerIndex}
                      headIndex={activeHeadIndex}
                      version={heatmapVersion}
                    />
                  ) : attentionViewMode === 'layer' ? (
                    <LayerOverview
                      layerHeatmaps={heatmapsRef.current[activeLayerIndex]}
                      heads={totalHeads}
                      version={heatmapVersion}
                      selectedHeadIndex={activeHeadIndex}
                      onPickHead={(headIndex) => {
                        setSelectedHeadIndex(headIndex);
                        setAttentionViewMode('single');
                      }}
                    />
                  ) : (
                    <AttentionOverview
                      heatmaps={heatmapsRef.current}
                      layers={totalLayers}
                      heads={totalHeads}
                      seqLen={Math.max(seqLen, seqLenRef.current)}
                      version={heatmapVersion}
                      selectedLayerIndex={activeLayerIndex}
                      onPick={(layerIndex) => {
                        setSelectedLayerIndex(layerIndex);
                        setSelectedHeadIndex(0);
                        setAttentionViewMode('layer');
                      }}
                    />
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* LAYER 4: AI Answer Panel */}
      {aiAnswer !== null && (
        <Card className="pointer-events-auto absolute top-20 left-6 z-20 flex max-h-[calc(100vh-160px)] w-90 flex-col overflow-hidden border-white/10 bg-black/70 backdrop-blur-xl">
          <CardHeader className="relative pb-0">
            <CardTitle className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
              AI Answer
            </CardTitle>
            <Button
              variant="ghost"
              size="icon-xs"
              className="absolute top-3 right-3 text-white/70 hover:text-white"
              onClick={() => {
                setAiAnswer(null);
                setAiSources([]);
                setAiSourceNodes([]);
                setActiveNodes([]);
              }}
            >
              ✕
            </Button>
          </CardHeader>
          <div className="min-h-0 flex-1 overflow-y-auto">
            <CardContent className="pb-4">
              <p className="text-sm leading-relaxed text-white/90">{aiAnswer}</p>

              {aiSources.length > 0 && (
                <>
                  <Separator className="my-3 bg-white/10" />
                  <h4 className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
                    Sources ({aiSources.length})
                  </h4>
                  <div className="flex flex-col gap-2">
                    {aiSources.map((src, i) => {
                      const matchedNode = nodes.find((n) => n.text === src.text);
                      return (
                        <div
                          key={i}
                          onClick={() => {
                            if (matchedNode) {
                              const clusterIds = new Set<string>([matchedNode.id]);
                              for (const edge of edges) {
                                if (edge.sourceId === matchedNode.id) clusterIds.add(edge.targetId);
                                if (edge.targetId === matchedNode.id) clusterIds.add(edge.sourceId);
                              }
                              const { activeNodeIds } = useAppStore.getState();
                              const mergedIds = new Set<string>([...activeNodeIds, ...clusterIds]);
                              setActiveNodes([...mergedIds]);
                              setSelectedNode(matchedNode.id);
                            }
                          }}
                          className={`rounded-lg p-2.5 text-xs transition-colors ${
                            matchedNode
                              ? 'cursor-pointer bg-white/5 border border-purple-500/30 hover:bg-white/10'
                              : 'bg-white/5 border border-transparent'
                          }`}
                        >
                          <div className="mb-1 flex items-center gap-1.5 text-muted-foreground">
                            {matchedNode && (
                              <div
                                className="size-1.5 shrink-0 rounded-full"
                                style={{ background: matchedNode.color }}
                              />
                            )}
                            <span>{src.metadata.source}</span>
                            {matchedNode && (
                              <span className="opacity-70">— {matchedNode.label}</span>
                            )}
                          </div>
                          <p className="leading-relaxed text-white/80">
                            {src.text.length > 200 ? src.text.slice(0, 200) + '...' : src.text}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </CardContent>
          </div>
        </Card>
      )}

      {/* LAYER 5: Node Detail Side Panel */}
      {(() => {
        const selectedNode = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) : null;
        const isOpen = selectedNode !== null && selectedNode !== undefined;
        const neighbors = isOpen
          ? edges
              .filter((e) => e.sourceId === selectedNodeId || e.targetId === selectedNodeId)
              .map((e) => {
                const neighborId = e.sourceId === selectedNodeId ? e.targetId : e.sourceId;
                return nodes.find((n) => n.id === neighborId);
              })
              .filter(Boolean)
          : [];

        return (
          <div
            className={`pointer-events-auto absolute top-20 bottom-44 right-0 z-20 w-80 bg-black/70 backdrop-blur-xl border-l border-white/10 text-white transition-transform duration-300 ease-in-out ${
              isOpen ? 'translate-x-0' : 'translate-x-full pointer-events-none'
            }`}
          >
            {isOpen && (
              <ScrollArea className="h-full">
                <div className="p-6">
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    className="absolute top-3 right-3 text-white/70 hover:text-white"
                    onClick={() => {
                      setSelectedNode(null);
                      setActiveNodes([]);
                    }}
                  >
                    ✕
                  </Button>

                  <div className="mb-4 flex items-center gap-2.5">
                    <div
                      className="size-3 shrink-0 rounded-full"
                      style={{ background: selectedNode.color }}
                    />
                    <h2 className="text-xl font-semibold">{selectedNode.label}</h2>
                  </div>

                  <div className="mb-4 space-y-0.5 text-xs text-muted-foreground">
                    <div>Color: {selectedNode.color}</div>
                    <div>
                      Position: ({selectedNode.position[0].toFixed(2)},{' '}
                      {selectedNode.position[1].toFixed(2)},{' '}
                      {selectedNode.position[2].toFixed(2)})
                    </div>
                  </div>

                  {selectedNode.text && (
                    <div className="mb-4">
                      <h3 className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
                        Content
                      </h3>
                      <p className="text-sm leading-relaxed text-white/80">{selectedNode.text}</p>
                    </div>
                  )}

                  <Separator className="my-3 bg-white/10" />

                  <h3 className="mb-2 text-xs font-medium uppercase tracking-widest text-muted-foreground">
                    Connections ({neighbors.length})
                  </h3>
                  {neighbors.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No connections</p>
                  ) : (
                    <div className="flex flex-col gap-1">
                      {neighbors.map((neighbor) => (
                        <div
                          key={neighbor!.id}
                          onClick={() => setSelectedNode(neighbor!.id)}
                          className="flex cursor-pointer items-center gap-2 rounded-lg bg-white/5 px-2.5 py-2 text-sm hover:bg-white/10 transition-colors"
                        >
                          <div
                            className="size-2 shrink-0 rounded-full"
                            style={{ background: neighbor!.color }}
                          />
                          {neighbor!.label}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </ScrollArea>
            )}
          </div>
        );
      })()}
    </div>
  );
}
