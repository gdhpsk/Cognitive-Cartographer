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
import { Switch } from '@/components/ui/switch';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import Markdown from 'react-markdown';
import { Textarea } from '@/components/ui/textarea';
import { SpotlightCard } from '@/components/ui/spotlight-card';
import { TextGenerateEffect } from '@/components/ui/text-generate';
import { GlowingBorder } from '@/components/ui/glowing-border';
import { fetchModelPolicy, chooseLlmModel } from '@/lib/llm-api';

const MIN_SCREEN_WIDTH = 1200;
const DEFAULT_MAX_PDF_SIZE_MB = 0;

function formatMegabytes(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

interface QuerySource {
  text: string;
  metadata: { chat_id: number; length: number; source: string };
}

type HeadHeatmap = {
  size: number;
  values: Float32Array;
};

type WebglHeatmapRenderer = {
  canvas: HTMLCanvasElement;
  gl: WebGL2RenderingContext;
  program: WebGLProgram;
  texture: WebGLTexture;
  vertexBuffer: WebGLBuffer;
  aPosition: number;
  uData: WebGLUniformLocation;
  uScaleMax: WebGLUniformLocation;
  uWidthPx: WebGLUniformLocation;
  uHeightPx: WebGLUniformLocation;
  uUseLowerTriMask: WebGLUniformLocation;
  uShowHeadSeparators: WebGLUniformLocation;
  uTileSizePx: WebGLUniformLocation;
  textureWidth: number;
  textureHeight: number;
};

type WebglPaintOptions = {
  maxValue: number;
  useLowerTriMask: boolean;
  showHeadSeparators: boolean;
  tileSizePx: number;
};

let sharedWebglHeatmapRenderer: WebglHeatmapRenderer | null = null;

const HEATMAP_VERTEX_SHADER = `
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
  v_uv = (a_position + 1.0) * 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const HEATMAP_FRAGMENT_SHADER = `
precision highp float;

varying vec2 v_uv;
uniform sampler2D u_data;
uniform float u_scaleMax;
uniform float u_widthPx;
uniform float u_heightPx;
uniform float u_useLowerTriMask;
uniform float u_showHeadSeparators;
uniform float u_tileSizePx;

void main() {
  vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
  float value = max(texture2D(u_data, uv).r, 0.0);
  float normalized = clamp(value / max(u_scaleMax, 1e-6), 0.0, 1.0);

  if (u_useLowerTriMask > 0.5) {
    float xPx = floor(uv.x * u_widthPx);
    float yPx = floor(uv.y * u_heightPx);
    if (xPx > yPx) {
      normalized = 0.0;
    }
  }

  if (u_showHeadSeparators > 0.5 && u_tileSizePx > 1.0) {
    float xPx = floor(uv.x * u_widthPx);
    if (xPx > 0.0 && mod(xPx + 1.0, u_tileSizePx) < 1.0) {
      gl_FragColor = vec4(100.0 / 255.0, 100.0 / 255.0, 100.0 / 255.0, 1.0);
      return;
    }
  }

  gl_FragColor = vec4(0.0, normalized, normalized, 1.0);
}
`;

function toAbsoluteIntensity(value: number, maxValue: number): number {
  if (!Number.isFinite(value)) return 0;
  const scaled = (Math.max(value, 0) / Math.max(maxValue, 1e-6)) * 255;
  if (scaled <= 0) return 0;
  if (scaled >= 255) return 255;
  return Math.round(scaled);
}

function getPositivePeak(values: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (Number.isFinite(value) && value > peak) {
      peak = value;
    }
  }
  return peak > 0 ? peak : 1;
}

function resolveScaleMax(scaleMax: number | undefined, fallbackValues: Float32Array): number {
  if (typeof scaleMax === 'number' && Number.isFinite(scaleMax) && scaleMax > 0) {
    return scaleMax;
  }
  return getPositivePeak(fallbackValues);
}

function compileWebglShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createWebglProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string
): WebGLProgram | null {
  const vertexShader = compileWebglShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileWebglShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) {
    if (vertexShader) gl.deleteShader(vertexShader);
    if (fragmentShader) gl.deleteShader(fragmentShader);
    return null;
  }

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    return null;
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

function createWebglHeatmapRenderer(): WebglHeatmapRenderer | null {
  const internalCanvas = document.createElement('canvas');
  const gl = internalCanvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) return null;

  const program = createWebglProgram(gl, HEATMAP_VERTEX_SHADER, HEATMAP_FRAGMENT_SHADER);
  if (!program) return null;

  const texture = gl.createTexture();
  const vertexBuffer = gl.createBuffer();
  if (!texture || !vertexBuffer) {
    if (texture) gl.deleteTexture(texture);
    if (vertexBuffer) gl.deleteBuffer(vertexBuffer);
    gl.deleteProgram(program);
    return null;
  }

  const aPosition = gl.getAttribLocation(program, 'a_position');
  const uData = gl.getUniformLocation(program, 'u_data');
  const uScaleMax = gl.getUniformLocation(program, 'u_scaleMax');
  const uWidthPx = gl.getUniformLocation(program, 'u_widthPx');
  const uHeightPx = gl.getUniformLocation(program, 'u_heightPx');
  const uUseLowerTriMask = gl.getUniformLocation(program, 'u_useLowerTriMask');
  const uShowHeadSeparators = gl.getUniformLocation(program, 'u_showHeadSeparators');
  const uTileSizePx = gl.getUniformLocation(program, 'u_tileSizePx');

  if (
    aPosition < 0 ||
    !uData ||
    !uScaleMax ||
    !uWidthPx ||
    !uHeightPx ||
    !uUseLowerTriMask ||
    !uShowHeadSeparators ||
    !uTileSizePx
  ) {
    gl.deleteTexture(texture);
    gl.deleteBuffer(vertexBuffer);
    gl.deleteProgram(program);
    return null;
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([
      -1, -1,
      1, -1,
      -1, 1,
      1, 1,
    ]),
    gl.STATIC_DRAW
  );

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);

  const renderer: WebglHeatmapRenderer = {
    canvas: internalCanvas,
    gl,
    program,
    texture,
    vertexBuffer,
    aPosition,
    uData,
    uScaleMax,
    uWidthPx,
    uHeightPx,
    uUseLowerTriMask,
    uShowHeadSeparators,
    uTileSizePx,
    textureWidth: 0,
    textureHeight: 0,
  };

  return renderer;
}

function getSharedWebglHeatmapRenderer(): WebglHeatmapRenderer | null {
  if (sharedWebglHeatmapRenderer) return sharedWebglHeatmapRenderer;
  sharedWebglHeatmapRenderer = createWebglHeatmapRenderer();
  return sharedWebglHeatmapRenderer;
}

function paintFloatBufferWebgl(
  canvas: HTMLCanvasElement,
  values: Float32Array,
  width: number,
  height: number,
  options: WebglPaintOptions
): boolean {
  const renderer = getSharedWebglHeatmapRenderer();
  if (!renderer) return false;

  if (renderer.canvas.width !== width) renderer.canvas.width = width;
  if (renderer.canvas.height !== height) renderer.canvas.height = height;

  const { gl } = renderer;
  gl.viewport(0, 0, width, height);
  gl.useProgram(renderer.program);

  gl.bindBuffer(gl.ARRAY_BUFFER, renderer.vertexBuffer);
  gl.enableVertexAttribArray(renderer.aPosition);
  gl.vertexAttribPointer(renderer.aPosition, 2, gl.FLOAT, false, 0, 0);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, renderer.texture);

  if (renderer.textureWidth !== width || renderer.textureHeight !== height) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, values);
    renderer.textureWidth = width;
    renderer.textureHeight = height;
  } else {
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RED, gl.FLOAT, values);
  }

  if (gl.getError() !== gl.NO_ERROR) {
    sharedWebglHeatmapRenderer = null;
    return false;
  }

  gl.uniform1i(renderer.uData, 0);
  gl.uniform1f(renderer.uScaleMax, Math.max(options.maxValue, 1e-6));
  gl.uniform1f(renderer.uWidthPx, width);
  gl.uniform1f(renderer.uHeightPx, height);
  gl.uniform1f(renderer.uUseLowerTriMask, options.useLowerTriMask ? 1 : 0);
  gl.uniform1f(renderer.uShowHeadSeparators, options.showHeadSeparators ? 1 : 0);
  gl.uniform1f(renderer.uTileSizePx, Math.max(1, options.tileSizePx));
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  if (gl.getError() !== gl.NO_ERROR) {
    sharedWebglHeatmapRenderer = null;
    return false;
  }

  const targetCtx = canvas.getContext('2d');
  if (!targetCtx) return false;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  targetCtx.clearRect(0, 0, width, height);
  targetCtx.drawImage(renderer.canvas, 0, 0, width, height);

  return true;
}

function toFiniteNumber(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
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
      heatmap.values[row * size + col] = toFiniteNumber(rowValues[col]);
    }
  }
}

function writeRow(heatmap: HeadHeatmap, rowValues: unknown[], rowIndex: number): void {
  const size = heatmap.size;
  if (size <= 0) return;

  const safeRow = Math.max(0, Math.min(size - 1, rowIndex));
  const maxCol = Math.min(size, rowValues.length, safeRow + 1);
  for (let col = 0; col < maxCol; col++) {
    heatmap.values[safeRow * size + col] = toFiniteNumber(rowValues[col]);
  }
}

function buildLayerAtlasValues(
  layerHeatmaps: HeadHeatmap[] | undefined,
  heads: number,
  seqLen: number
): { values: Float32Array; width: number; height: number; tileSize: number; safeHeads: number } {
  const safeHeads = Math.max(1, heads);
  const tileSize = Math.max(1, seqLen);
  const width = safeHeads * tileSize;
  const height = tileSize;
  const values = new Float32Array(width * height);

  for (let headIndex = 0; headIndex < safeHeads; headIndex++) {
    const heatmap = layerHeatmaps?.[headIndex] ?? null;
    const sourceSize = heatmap?.size ?? 0;

    for (let row = 0; row < tileSize; row++) {
      const sourceRow = row < sourceSize ? row : -1;
      for (let col = 0; col < tileSize; col++) {
        if (sourceRow >= 0 && col < sourceSize && col <= row && heatmap) {
          const x = headIndex * tileSize + col;
          const y = row;
          values[y * width + x] = heatmap.values[sourceRow * sourceSize + col];
        }
      }
    }
  }

  return { values, width, height, tileSize, safeHeads };
}

function paintHeatmapFallback2d(
  canvas: HTMLCanvasElement,
  heatmap: HeadHeatmap | null,
  scaleMax?: number
): void {
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
  const maxValue = resolveScaleMax(scaleMax, values);

  if (canvas.width !== size) canvas.width = size;
  if (canvas.height !== size) canvas.height = size;

  const image = ctx.createImageData(size, size);
  const data = image.data;

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const pixelIndex = (row * size + col) * 4;
      const value = col <= row ? values[row * size + col] : 0;
      const intensity = toAbsoluteIntensity(value, maxValue);

      data[pixelIndex] = 0;
      data[pixelIndex + 1] = intensity;
      data[pixelIndex + 2] = intensity;
      data[pixelIndex + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function paintLayerAtlasFallback2d(
  canvas: HTMLCanvasElement,
  atlasValues: Float32Array,
  width: number,
  height: number,
  safeHeads: number,
  tileSize: number,
  scaleMax?: number
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  const maxValue = resolveScaleMax(scaleMax, atlasValues);
  const image = ctx.createImageData(width, height);
  const data = image.data;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = (y * width + x) * 4;
      const value = atlasValues[y * width + x];
      const intensity = toAbsoluteIntensity(value, maxValue);

      data[pixelIndex] = 0;
      data[pixelIndex + 1] = intensity;
      data[pixelIndex + 2] = intensity;
      data[pixelIndex + 3] = 255;
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

function paintHeatmap(
  canvas: HTMLCanvasElement,
  heatmap: HeadHeatmap | null,
  preferWebgl = true,
  scaleMax?: number
): void {
  if (!preferWebgl) {
    paintHeatmapFallback2d(canvas, heatmap, scaleMax);
    return;
  }

  if (!heatmap) {
    const empty = new Float32Array(1);
    const painted = paintFloatBufferWebgl(canvas, empty, 1, 1, {
      maxValue: 1,
      useLowerTriMask: false,
      showHeadSeparators: false,
      tileSizePx: 1,
    });
    if (!painted) {
      paintHeatmapFallback2d(canvas, null, scaleMax);
    }
    return;
  }

  const { size, values } = heatmap;
  const maxValue = resolveScaleMax(scaleMax, values);
  const painted = paintFloatBufferWebgl(canvas, values, size, size, {
    maxValue,
    useLowerTriMask: true,
    showHeadSeparators: false,
    tileSizePx: size,
  });
  if (!painted) {
    paintHeatmapFallback2d(canvas, heatmap, scaleMax);
  }
}

function paintLayerAtlas(
  canvas: HTMLCanvasElement,
  layerHeatmaps: HeadHeatmap[] | undefined,
  heads: number,
  seqLen: number,
  preferWebgl = true,
  scaleMax?: number
): void {
  const atlas = buildLayerAtlasValues(layerHeatmaps, heads, seqLen);
  if (preferWebgl) {
    const maxValue = resolveScaleMax(scaleMax, atlas.values);
    const painted = paintFloatBufferWebgl(canvas, atlas.values, atlas.width, atlas.height, {
      maxValue,
      useLowerTriMask: false,
      showHeadSeparators: true,
      tileSizePx: atlas.tileSize,
    });
    if (painted) return;
  }

  paintLayerAtlasFallback2d(
    canvas,
    atlas.values,
    atlas.width,
    atlas.height,
    atlas.safeHeads,
    atlas.tileSize,
    scaleMax
  );
}

function AttentionOverview({
  heatmaps,
  layers,
  heads,
  seqLen,
  version,
  scaleMax,
  onPick,
  selectedLayerIndex,
}: {
  heatmaps: HeadHeatmap[][];
  layers: number;
  heads: number;
  seqLen: number;
  version: number;
  scaleMax?: number;
  onPick?: (layerIndex: number) => void;
  selectedLayerIndex?: number;
}) {
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([]);
  const [hoverLabel, setHoverLabel] = useState('Hover a layer plane to inspect');
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
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
    const el = containerRef.current;
    if (!el) return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      setContainerSize({
        width: Math.max(1, Math.round(rect.width)),
        height: Math.max(1, Math.round(rect.height)),
      });
    };

    updateSize();

    const observer = new ResizeObserver(() => {
      updateSize();
    });
    observer.observe(el);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    for (let layerIndex = 0; layerIndex < safeLayers; layerIndex++) {
      const canvas = canvasRefs.current[layerIndex];
      if (!canvas) continue;
      paintLayerAtlas(canvas, heatmaps[layerIndex], safeHeads, safeSeqLen, true, scaleMax);
    }
  }, [heatmaps, safeLayers, safeHeads, safeSeqLen, scaleMax, version]);

  const pinchRef = useRef<{ dist: number } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const clampZoom = (z: number) => Math.min(5, Math.max(0.2, z));

    // scroll-wheel zoom
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setZoom((prev) => clampZoom(prev * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
    };

    // pinch-to-zoom
    const getTouchDist = (e: TouchEvent) => {
      const [a, b] = [e.touches[0], e.touches[1]];
      return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    };
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        e.preventDefault();
        pinchRef.current = { dist: getTouchDist(e) };
      }
    };
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && pinchRef.current) {
        e.preventDefault();
        const newDist = getTouchDist(e);
        const scale = newDist / pinchRef.current.dist;
        pinchRef.current.dist = newDist;
        setZoom((prev) => clampZoom(prev * scale));
      }
    };
    const onTouchEnd = () => {
      pinchRef.current = null;
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('touchstart', onTouchStart, { passive: false });
    el.addEventListener('touchmove', onTouchMove, { passive: false });
    el.addEventListener('touchend', onTouchEnd);
    el.addEventListener('touchcancel', onTouchEnd);
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('touchstart', onTouchStart);
      el.removeEventListener('touchmove', onTouchMove);
      el.removeEventListener('touchend', onTouchEnd);
      el.removeEventListener('touchcancel', onTouchEnd);
    };
  }, []);

  useEffect(() => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
    setIsDragging(false);
    dragStateRef.current.active = false;
    dragStateRef.current.moved = false;
  }, [safeLayers, safeHeads]);

  const minViewportDim = Math.max(1, Math.min(containerSize.width || 1, containerSize.height || 1));
  const viewportScale = Math.max(0.75, Math.min(1.6, minViewportDim / 900));
  const perspectivePx = Math.round(Math.max(1300, Math.min(3200, minViewportDim * 2.1)));

  const baseDepthStep = safeLayers > 1 ? Math.max(24, Math.min(124, Math.floor(1450 / safeLayers))) : 0;
  const baseYStep = safeLayers > 1 ? Math.max(9, Math.min(34, Math.floor(560 / safeLayers))) : 0;
  const depthStep = safeLayers > 1
    ? Math.round(baseDepthStep * (0.9 + (viewportScale - 1) * 0.35))
    : 0;
  const yStep = safeLayers > 1
    ? Math.round(baseYStep * (0.92 + (viewportScale - 1) * 0.22))
    : 0;

  const baseStageScale = Math.max(0.22, Math.min(0.9, 1.08 - safeLayers * 0.026));
  const stageScale = Math.max(0.22, Math.min(1.08, baseStageScale * (0.95 + (viewportScale - 1) * 0.25)));

  const stageWidthPx = Math.round(Math.max(920, Math.min((containerSize.width || 1200) * 1.22, 2500)));
  const stageHeightPx = Math.round(Math.max(640, Math.min((containerSize.height || 800) * 1.16, 1800)));
  const tiltX = Math.max(49, Math.min(58, 57 - (viewportScale - 1) * 4));
  const tiltZ = Math.max(-12, Math.min(-7, -10 + (viewportScale - 1) * 2.5));

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
        ref={containerRef}
        className={`absolute inset-0 flex items-center justify-center ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
        style={{ perspective: `${perspectivePx}px` }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onDoubleClick={() => { setPan({ x: 0, y: 0 }); setZoom(1); }}
      >
        <div
          className="relative"
          style={{
            width: `${stageWidthPx}px`,
            height: `${stageHeightPx}px`,
            transformStyle: 'preserve-3d',
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${stageScale * zoom}) rotateX(${tiltX}deg) rotateZ(${tiltZ}deg)`,
            willChange: 'transform',
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

      {/* Zoom slider */}
      <div className="pointer-events-auto absolute bottom-3 right-3 flex items-center gap-2 rounded-lg bg-black/70 px-3 py-1.5 backdrop-blur-sm">
        <button
          type="button"
          className="text-xs text-muted-foreground hover:text-white transition-colors"
          onClick={() => setZoom((prev) => Math.max(0.2, prev / 1.3))}
        >
          −
        </button>
        <Slider
          min={0.2}
          max={5}
          step={0.05}
          value={[zoom]}
          className="w-24"
          onValueChange={(val) => {
            const v = typeof val === 'number' ? val : Array.isArray(val) ? val[0] : undefined;
            if (typeof v === 'number') setZoom(v);
          }}
        />
        <button
          type="button"
          className="text-xs text-muted-foreground hover:text-white transition-colors"
          onClick={() => setZoom((prev) => Math.min(5, prev * 1.3))}
        >
          +
        </button>
        <span className="ml-1 min-w-8 text-right text-[10px] text-muted-foreground">{Math.round(zoom * 100)}%</span>
      </div>
    </div>
  );
}

function SingleHeadView({
  heatmap,
  layerIndex,
  headIndex,
  scaleMax,
  version,
  tokens,
}: {
  heatmap: HeadHeatmap | null;
  layerIndex: number;
  headIndex: number;
  scaleMax?: number;
  version: number;
  tokens?: string[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; row: number; col: number; value: number } | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    paintHeatmap(canvasRef.current, heatmap, true, scaleMax);
  }, [heatmap, scaleMax, version]);

  const size = heatmap?.size ?? 1;

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!heatmap) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * size);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * size);
    if (row < 0 || row >= size || col < 0 || col >= size || col > row) {
      setTooltip(null);
      return;
    }
    const value = heatmap.values[row * size + col];
    setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, row, col, value });
  };

  const toToken = tokens?.[tooltip?.row ?? -1];
  const fromToken = tokens?.[tooltip?.col ?? -1];

  return (
    <div className="relative flex h-full w-full items-center justify-center overflow-hidden p-2">
      <div className="relative h-full w-full max-h-[88vh] max-w-[88vh]">
        <canvas
          ref={canvasRef}
          width={size}
          height={size}
          className="h-full w-full rounded border border-cyan-300/40 bg-black [image-rendering:pixelated]"
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={() => setTooltip(null)}
        />
        <div className="pointer-events-none absolute left-2 top-2 rounded bg-black/65 px-2 py-1 text-[11px] text-cyan-200">
          Layer {layerIndex + 1} • Head {headIndex + 1}
        </div>
        {tooltip && (
          <div
            className="pointer-events-none absolute z-50 rounded bg-black/85 px-2 py-1.5 text-[10px] leading-snug text-white shadow-lg backdrop-blur-sm border border-white/10"
            style={{ left: tooltip.x + 12, top: tooltip.y - 8, maxWidth: 220 }}
          >
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-cyan-300">{fromToken ?? `[${tooltip.col}]`}</span>
              <span className="text-white/40">→</span>
              <span className="font-mono text-cyan-300">{toToken ?? `[${tooltip.row}]`}</span>
            </div>
            <div className="mt-0.5 text-white/60">
              intensity: <span className="text-cyan-200 font-mono">{tooltip.value.toFixed(4)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function LayerOverview({
  layerHeatmaps,
  heads,
  scaleMax,
  version,
  selectedHeadIndex,
  onPickHead,
  tokens,
}: {
  layerHeatmaps: HeadHeatmap[] | undefined;
  heads: number;
  scaleMax?: number;
  version: number;
  selectedHeadIndex: number;
  onPickHead?: (headIndex: number) => void;
  tokens?: string[];
}) {
  const canvasRefs = useRef<Array<HTMLCanvasElement | null>>([]);
  const safeHeads = Math.max(1, heads);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; headIndex: number; row: number; col: number; value: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    for (let headIndex = 0; headIndex < safeHeads; headIndex++) {
      const canvas = canvasRefs.current[headIndex];
      if (!canvas) continue;
      paintHeatmap(canvas, layerHeatmaps?.[headIndex] ?? null, true, scaleMax);
    }
  }, [layerHeatmaps, safeHeads, scaleMax, version]);

  const columns =
    safeHeads >= 28 ? 8 :
    safeHeads >= 16 ? 6 :
    safeHeads >= 9 ? 5 :
    safeHeads >= 6 ? 4 :
    safeHeads;

  const handleHeadMouseMove = (e: React.MouseEvent<HTMLCanvasElement>, headIndex: number) => {
    const heatmap = layerHeatmaps?.[headIndex];
    if (!heatmap || !containerRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const containerRect = containerRef.current.getBoundingClientRect();
    const size = heatmap.size;
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * size);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * size);
    if (row < 0 || row >= size || col < 0 || col >= size || col > row) {
      setTooltip(null);
      return;
    }
    const value = heatmap.values[row * size + col];
    setTooltip({
      x: e.clientX - containerRect.left,
      y: e.clientY - containerRect.top,
      headIndex, row, col, value,
    });
  };

  const toToken = tokens?.[tooltip?.row ?? -1];
  const fromToken = tokens?.[tooltip?.col ?? -1];

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden p-2">
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
              onMouseMove={(e) => handleHeadMouseMove(e, headIndex)}
              onMouseLeave={() => setTooltip(null)}
            />
            <div className="pointer-events-none absolute left-1 top-1 rounded bg-black/70 px-1 py-0.5 text-[10px] text-cyan-200">
              H{headIndex + 1}
            </div>
          </button>
        ))}
      </div>
      {tooltip && (
        <div
          className="pointer-events-none absolute z-50 rounded bg-black/85 px-2 py-1.5 text-[10px] leading-snug text-white shadow-lg backdrop-blur-sm border border-white/10"
          style={{ left: tooltip.x + 12, top: tooltip.y - 8, maxWidth: 220 }}
        >
          <div className="text-white/50 mb-0.5">Head {tooltip.headIndex + 1}</div>
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-cyan-300">{fromToken ?? `[${tooltip.col}]`}</span>
            <span className="text-white/40">→</span>
            <span className="font-mono text-cyan-300">{toToken ?? `[${tooltip.row}]`}</span>
          </div>
          <div className="mt-0.5 text-white/60">
            intensity: <span className="text-cyan-200 font-mono">{tooltip.value.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [apiHost, setApiHost] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('cog-cart-api-host') || '';
    }
    return '';
  });
  const [apiHostInput, setApiHostInput] = useState(apiHost);
  const [queryText, setQueryText] = useState('');
  const [aiQuery, setAiQuery] = useState('');
  const [aiK, setAiK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(200);
  const [isAttentionPopoverOpen, setIsAttentionPopoverOpen] = useState(false);
  const [attentionViewMode, setAttentionViewMode] = useState<'stack' | 'layer' | 'single'>('stack');
  const [selectedLayerIndex, setSelectedLayerIndex] = useState(0);
  const [selectedHeadIndex, setSelectedHeadIndex] = useState(0);
  const [showAttentionMeta, setShowAttentionMeta] = useState(false);
  const [useGlobalScale, setUseGlobalScale] = useState(true);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModelOption, setSelectedModelOption] = useState('');
  const [customModelName, setCustomModelName] = useState('');
  const [allowCustomHfModels, setAllowCustomHfModels] = useState(false);
  const [isCustomModelInputEnabled, setIsCustomModelInputEnabled] = useState(false);
  const [activeModelName, setActiveModelName] = useState('');
  const [modelStatus, setModelStatus] = useState('');
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [maxPdfSizeMb, setMaxPdfSizeMb] = useState<number>(DEFAULT_MAX_PDF_SIZE_MB);
  const [maxSeqLen, setMaxSeqLen] = useState<number>(0);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [aiSources, setAiSources] = useState<QuerySource[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [queuePosition, setQueuePosition] = useState<number | null>(null);
  const [attentionStatus, setAttentionStatus] = useState('Idle');
  const [promptTokens, setPromptTokens] = useState<string[]>([]);
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);
  const [seqLen, setSeqLen] = useState(0);
  const [totalLayers, setTotalLayers] = useState(0);
  const [totalHeads, setTotalHeads] = useState(0);
  const [heatmapVersion, setHeatmapVersion] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [uploadErrorMessage, setUploadErrorMessage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [hasUploadedPdf, setHasUploadedPdf] = useState(false);
  const [sessionLost, setSessionLost] = useState(false);
  const [pendingUploadFile, setPendingUploadFile] = useState<File | null>(null);
  const [isScreenWideEnough, setIsScreenWideEnough] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);
  const [analysisPrompt, setAnalysisPrompt] = useState('Analyze this attention heatmap. Describe what patterns you see in the attention weights, which tokens attend to which, and any notable structure.');
  const [queryCardOffset, setQueryCardOffset] = useState({ x: 0, y: 0 });
  const [isQueryCardDragging, setIsQueryCardDragging] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const heatmapsRef = useRef<HeadHeatmap[][]>([]);
  const seqLenRef = useRef(0);
  const totalLayersRef = useRef(0);
  const totalHeadsRef = useRef(0);
  const queryCardDragRef = useRef({
    active: false,
    pointerId: -1,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
  });
  const { nodes, edges, isLoading, setLoading, setActiveNodes, setAiSourceNodes, selectedNodeId, setSelectedNode, loadGraph, sessionId, setSessionId, setUploadSocket } = useAppStore();

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
    setAnalysisResult(null);
  }, []);

  const handleAnalyzeHeatmap = useCallback(async () => {
    if (isAnalyzing || !apiHost) return;

    // Find the visible heatmap canvas inside the attention popover
    const popover = document.querySelector('[data-slot="attention-popover"]');
    if (!popover) return;

    // Grab all heatmap canvases currently rendered in the visualization area
    const canvases = popover.querySelectorAll<HTMLCanvasElement>('[data-slot="attention-viz"] canvas');
    if (canvases.length === 0) return;

    // Composite all visible canvases into one image
    const compositeCanvas = document.createElement('canvas');
    const ctx = compositeCanvas.getContext('2d');
    if (!ctx) return;

    if (canvases.length === 1) {
      // Single canvas — just export it directly
      const src = canvases[0];
      compositeCanvas.width = src.width;
      compositeCanvas.height = src.height;
      ctx.drawImage(src, 0, 0);
    } else {
      // Multiple canvases (layer/grid view) — tile them into a composite
      const gap = 2;
      const cols = Math.min(canvases.length, 6);
      const rows = Math.ceil(canvases.length / cols);
      const tileW = canvases[0].width || 64;
      const tileH = canvases[0].height || 64;
      compositeCanvas.width = cols * tileW + (cols - 1) * gap;
      compositeCanvas.height = rows * tileH + (rows - 1) * gap;
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);
      canvases.forEach((c, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        ctx.drawImage(c, col * (tileW + gap), row * (tileH + gap), tileW, tileH);
      });
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        compositeCanvas.toBlob(resolve, 'image/png')
      );
      if (!blob) {
        setAnalysisResult('Failed to capture heatmap image.');
        setIsAnalyzing(false);
        return;
      }

      // Build context about the current view
      const contextParts: string[] = [];

      // View mode and selection
      if (attentionViewMode === 'single') {
        contextParts.push(`Viewing: Single head — Layer ${selectedLayerIndex + 1}/${totalLayers}, Head ${selectedHeadIndex + 1}/${totalHeads}`);
      } else if (attentionViewMode === 'layer') {
        contextParts.push(`Viewing: All ${totalHeads} heads of Layer ${selectedLayerIndex + 1}/${totalLayers}`);
      } else {
        contextParts.push(`Viewing: 3D stack of all ${totalLayers} layers × ${totalHeads} heads`);
      }

      contextParts.push(`Sequence length: ${seqLen} tokens`);
      contextParts.push(`Scale: ${useGlobalScale ? 'global (0–1)' : 'local (per-head max)'}`);

      // Token list
      if (promptTokens.length > 0) {
        contextParts.push(`\nToken sequence (${promptTokens.length} tokens):\n${promptTokens.map((t, i) => `  [${i}] "${t}"`).join('\n')}`);
      }

      // Generated tokens
      if (generatedTokens.length > 0) {
        contextParts.push(`\nGenerated tokens (${generatedTokens.length}): ${generatedTokens.join(' ')}`);
      }

      // Heatmap statistics for current view
      if (attentionViewMode === 'single') {
        const heatmap = heatmapsRef.current[selectedLayerIndex]?.[selectedHeadIndex];
        if (heatmap) {
          const peak = getPositivePeak(heatmap.values);
          let sum = 0;
          let count = 0;
          for (let r = 0; r < heatmap.size; r++) {
            for (let c = 0; c <= r; c++) {
              const v = heatmap.values[r * heatmap.size + c];
              if (Number.isFinite(v) && v > 0) { sum += v; count++; }
            }
          }
          contextParts.push(`\nHeatmap stats: peak=${peak.toFixed(4)}, mean=${count > 0 ? (sum / count).toFixed(4) : '0'}, active_cells=${count}`);
        }
      }

      const enrichedPrompt = `${contextParts.join('\n')}\n\n---\nUser prompt: ${analysisPrompt}`;

      const formData = new FormData();
      formData.append('image', blob, 'attention_heatmap.png');
      formData.append('prompt', enrichedPrompt);

      const res = await fetch(`${API_BASE}/image_analysis`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (typeof data?.analysis === 'string') {
        setAnalysisResult(data.analysis);
      } else {
        setAnalysisResult('No analysis returned from the server.');
      }
    } catch {
      setAnalysisResult('Failed to connect to analysis service.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, analysisPrompt]);

  const handleAiQuery = useCallback(() => {
    if (!aiQuery.trim() || isQuerying) return;

    if (!apiHost) {
      setAiAnswer('API host is not configured. Set it in the setup screen.');
      return;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsQuerying(true);
    setQueuePosition(null);
    setIsAttentionPopoverOpen(true);
    setAttentionStatus('Connecting to /ws/attention...');
    setAiAnswer(null);
    setAiSources([]);
    resetAttentionState();

    const ws = new WebSocket(`${WS_BASE}/ws/attention`);
    wsRef.current = ws;
    let completed = false;

    ws.onopen = () => {
      setAttentionStatus('Connected. Streaming attention heads...');
      ws.send(JSON.stringify({ query: aiQuery.trim(), k: aiK, max_tokens: maxTokens, session_id: useAppStore.getState().sessionId }));
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

        if (eventName === 'queued' && isRecord(data)) {
          const pos = typeof data.position === 'number' ? data.position : null;
          setQueuePosition(pos);
          setAttentionStatus(typeof data.message === 'string' ? data.message : 'Queued — waiting for model...');
          return;
        }

        if (eventName === 'tokens' && isRecord(data)) {
          setQueuePosition(null);
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
          setAttentionStatus('Answer received. Generation complete.');
          completed = true;
          setIsQuerying(false);
          ws.close();
          return;
        }

        if (eventName === 'error') {
          const errorMsg = typeof data === 'string' ? data : (isRecord(data) && typeof data.message === 'string' ? data.message : 'An error occurred.');
          setQueuePosition(null);
          setAiAnswer(errorMsg);
          setAttentionStatus('Error from server.');
          completed = true;
          setIsQuerying(false);
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
          ws.close();
        }
      } catch {
        setAttentionStatus('Received malformed websocket payload.');
      }
    };

    ws.onerror = () => {
      setQueuePosition(null);
      setAiAnswer('Failed to connect to attention service.');
      setAttentionStatus('WebSocket error while streaming attention.');
      setIsQuerying(false);
      setIsAttentionPopoverOpen(false);
      resetAttentionState();
    };

    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
      if (!completed) {
        setQueuePosition(null);
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

  const isLocalhost = /^(localhost|127\.0\.0\.1)(:\d+)?$/.test(apiHost);
  const API_BASE = `${isLocalhost ? 'http' : 'https'}://${apiHost}`;
  const WS_BASE = `${isLocalhost ? 'ws' : 'wss'}://${apiHost}`;

  const fetchAvailableModels = useCallback(async () => {
    if (!apiHost) {
      setModelStatus('API host is not configured. Set it in the setup screen.');
      return;
    }

    setIsLoadingModels(true);
    try {
      const {
        availableModels: modelsFromPolicy,
        allowCustomHfModels: allowCustom,
        maxPdfSizeMb: serverMaxPdfSizeMb,
        maxSeqLen: serverMaxSeqLen,
      } = await fetchModelPolicy(API_BASE);

      setAvailableModels(modelsFromPolicy);
      setAllowCustomHfModels(allowCustom);
      setMaxPdfSizeMb(serverMaxPdfSizeMb);
      setMaxSeqLen(serverMaxSeqLen);
      setIsCustomModelInputEnabled((prev) => (allowCustom ? prev : false));
      setSelectedModelOption((prev) => {
        if (prev && modelsFromPolicy.includes(prev)) return prev;
        return modelsFromPolicy[0] ?? '';
      });

      if (modelsFromPolicy.length === 0) {
        setModelStatus('No models returned by /aval_model.');
      } else {
        setModelStatus(
          allowCustom
            ? `Loaded ${modelsFromPolicy.length} models.`
            : `Loaded ${modelsFromPolicy.length} models. Custom HF models are disabled by the server.`
        );
      }
    } catch (error) {
      console.error('Failed to fetch available models:', error);
      setAllowCustomHfModels(false);
      setMaxSeqLen(0);
      setIsCustomModelInputEnabled(false);
      setModelStatus('Failed to load models.');
    } finally {
      setIsLoadingModels(false);
    }
  }, [API_BASE]);

  const handleChooseModel = useCallback(async () => {
    const trimmedCustomModelName = customModelName.trim();
    const trimmedSelectedModel = selectedModelOption.trim();
    const modelName =
      allowCustomHfModels && isCustomModelInputEnabled && trimmedCustomModelName.length > 0
        ? trimmedCustomModelName
        : trimmedSelectedModel;

    if (!modelName) {
      setModelStatus('Select a model from the dropdown or enter a custom HF model ID.');
      return;
    }
    if (!apiHost) {
      setModelStatus('API host is not configured. Set it in the setup screen.');
      return;
    }

    setIsSwitchingModel(true);
    try {
      const result = await chooseLlmModel(API_BASE, modelName);

      if (!result.ok) {
        if (result.statusCode === 403) {
          setModelStatus(
            result.detail ||
              'This model is blocked by server policy. Choose one of the available models or ask the server admin to enable custom HF models.'
          );
          void fetchAvailableModels();
          return;
        }

        setModelStatus(result.detail ? `Failed to switch model: ${result.detail}` : 'Failed to switch model.');
        return;
      }

      setActiveModelName(modelName);
      setModelStatus(`Using model: ${modelName}`);
    } catch (error) {
      console.error('Failed to switch model:', error);
      setModelStatus('Failed to switch model.');
    } finally {
      setIsSwitchingModel(false);
    }
  }, [API_BASE, allowCustomHfModels, customModelName, fetchAvailableModels, isCustomModelInputEnabled, selectedModelOption]);

  const handleLoadGraph = useCallback(async () => {
    const currentSessionId = useAppStore.getState().sessionId;
    if (!currentSessionId) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/graph?session_id=${encodeURIComponent(currentSessionId)}`);
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
  }, [API_BASE, loadGraph, setLoading]);

  const handleUploadFileSelection = useCallback((files: File[]) => {
    if (isUploading) return;

    const file = files?.[0];
    if (!file) {
      setPendingUploadFile(null);
      return;
    }

    if (maxPdfSizeMb > 0 && file.size > maxPdfSizeMb * 1024 * 1024) {
      const message = `This PDF is too large (${(file.size / (1024 * 1024)).toFixed(2)} MB). Maximum allowed size is ${formatMegabytes(maxPdfSizeMb)} MB.`;
      setHasUploadedPdf(false);
      setPendingUploadFile(null);
      setUploadErrorMessage(message);
      setUploadStatus(message);
      return;
    }

    setUploadErrorMessage(null);
    setHasUploadedPdf(false);
    setPendingUploadFile(file);
    setUploadStatus(`Ready to upload: ${file.name}`);
  }, [isUploading, maxPdfSizeMb]);

  const handleConfirmUpload = useCallback(() => {
    if (!pendingUploadFile || isUploading) return;

    if (maxPdfSizeMb > 0 && pendingUploadFile.size > maxPdfSizeMb * 1024 * 1024) {
      const message = `This PDF is too large (${(pendingUploadFile.size / (1024 * 1024)).toFixed(2)} MB). Maximum allowed size is ${formatMegabytes(maxPdfSizeMb)} MB.`;
      setPendingUploadFile(null);
      setUploadErrorMessage(message);
      setUploadStatus(message);
      return;
    }

    // Close any existing session socket — backend will clean up the old document.
    const existingSocket = useAppStore.getState().uploadSocket;
    if (existingSocket) {
      existingSocket.onclose = null;
      existingSocket.close();
      setUploadSocket(null);
    }
    setSessionId(null);
    setSessionLost(false);
    setHasUploadedPdf(false);
    setUploadErrorMessage(null);
    setIsUploading(true);
    setUploadStatus('Connecting to upload service...');

    const uploadWs = new WebSocket(`${WS_BASE}/ws/upload`);
    let sessionSaved = false;

    uploadWs.onopen = () => {
      setUploadErrorMessage(null);
      setUploadStatus('Connected. Uploading PDF bytes...');
      uploadWs.send(pendingUploadFile);
    };

    uploadWs.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        if (typeof parsed?.data === 'string') setUploadStatus(parsed.data);
        if (parsed?.event === 'done') {
          const sid = typeof parsed.session_id === 'string' ? parsed.session_id : null;
          sessionSaved = true;
          setSessionId(sid);
          setUploadSocket(uploadWs);
          setHasUploadedPdf(true);
          setPendingUploadFile(null);
          setIsUploading(false);
          void handleLoadGraph();
          // Do NOT close the socket — keeping it open preserves the server-side session.
        }
      } catch {
        if (typeof msg.data === 'string') setUploadStatus(msg.data);
      }
    };

    uploadWs.onerror = () => {
      const message = 'Upload failed. Check that the server is running.';
      setUploadErrorMessage(message);
      setUploadStatus(message);
      setHasUploadedPdf(false);
      setIsUploading(false);
    };

    uploadWs.onclose = () => {
      setIsUploading(false);
      if (sessionSaved) {
        // Unexpected close after the session was established — session is now gone.
        setSessionId(null);
        setUploadSocket(null);
        setHasUploadedPdf(false);
        setPendingUploadFile(pendingUploadFile);
        setUploadStatus(`Session closed. Ready to re-upload: ${pendingUploadFile.name}`);
        setAiAnswer(null);
        setAiSources([]);
        setSessionLost(true);
      }
    };
  }, [handleLoadGraph, isUploading, maxPdfSizeMb, pendingUploadFile, setSessionId, setUploadSocket]);

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
      const sock = useAppStore.getState().uploadSocket;
      if (sock) {
        sock.onclose = null;
        sock.close();
        useAppStore.getState().setUploadSocket(null);
      }
    };
  }, []);

  useEffect(() => {
    const handleBeforeUnload = () => {
      const sock = useAppStore.getState().uploadSocket;
      if (sock) sock.close();
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, []);

  useEffect(() => {
    void fetchAvailableModels();
  }, [fetchAvailableModels]);

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
  const selectedNode = selectedNodeId ? nodes.find((n) => n.id === selectedNodeId) ?? null : null;
  const trimmedCustomModelName = customModelName.trim();
  const pendingModelName =
    allowCustomHfModels && isCustomModelInputEnabled && trimmedCustomModelName.length > 0
      ? trimmedCustomModelName
      : selectedModelOption.trim();
  const attentionScaleMax = useGlobalScale ? 1 : undefined;

  const handleQueryCardDragStart = (event: React.PointerEvent<HTMLDivElement>) => {
    queryCardDragRef.current = {
      active: true,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: queryCardOffset.x,
      originY: queryCardOffset.y,
    };
    setIsQueryCardDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handleQueryCardDragMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const dragState = queryCardDragRef.current;
    if (!dragState.active || dragState.pointerId !== event.pointerId) return;

    const dx = event.clientX - dragState.startX;
    const dy = event.clientY - dragState.startY;
    setQueryCardOffset({ x: dragState.originX + dx, y: dragState.originY + dy });
  };

  const handleQueryCardDragEnd = (event: React.PointerEvent<HTMLDivElement>) => {
    const dragState = queryCardDragRef.current;
    if (!dragState.active || dragState.pointerId !== event.pointerId) return;

    queryCardDragRef.current.active = false;
    setIsQueryCardDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  if (!apiHost) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#050510] p-6 text-white">
        <SpotlightCard className="w-full max-w-md rounded-xl">
          <Card className="bg-black/60 backdrop-blur-xl border-white/10">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-white">Cognitive Cartographer</CardTitle>
              <p className="text-sm text-muted-foreground">
                Enter your backend API hostname to get started.
              </p>
            </CardHeader>
            <CardContent>
              <form
                className="flex flex-col gap-3"
                onSubmit={(e) => {
                  e.preventDefault();
                  const trimmed = apiHostInput.trim().replace(/^https?:\/\//, '').replace(/\/+$/, '');
                  if (!trimmed) return;
                  localStorage.setItem('cog-cart-api-host', trimmed);
                  setApiHost(trimmed);
                }}
              >
                <div>
                  <label className="mb-1.5 block text-xs font-medium uppercase tracking-widest text-muted-foreground">
                    API Host
                  </label>
                  <Input
                    placeholder="e.g. api.example.com"
                    value={apiHostInput}
                    onChange={(e) => setApiHostInput(e.target.value)}
                    className="bg-black/50 border-white/10 text-white placeholder:text-white/30"
                  />
                  <p className="mt-1.5 text-[11px] text-muted-foreground">
                    Hostname only, no protocol. HTTPS and WSS are used automatically.
                  </p>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    <a
                      href="https://github.com/gdhpsk/Cognitive-Cartographer"
                      target="_blank"
                      rel="noreferrer"
                      className="text-cyan-300 underline underline-offset-2 hover:text-cyan-200"
                    >
                      How to get started
                    </a>
                  </p>
                </div>
                <Button
                  type="submit"
                  disabled={!apiHostInput.trim()}
                  className="w-full bg-cyan-500 text-black hover:bg-cyan-400 disabled:bg-secondary disabled:text-muted-foreground"
                >
                  Connect
                </Button>
              </form>
            </CardContent>
          </Card>
        </SpotlightCard>
      </div>
    );
  }

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

      {/* No-session overlay on graph */}
      {!sessionId && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <p className="rounded-lg bg-black/50 px-4 py-2 text-sm text-white/40 backdrop-blur-sm">
            Upload a document to explore the knowledge graph.
          </p>
        </div>
      )}

      {/* Session-lost banner */}
      {sessionLost && (
        <div className="pointer-events-auto absolute left-1/2 top-4 z-50 -translate-x-1/2">
          <div className="flex items-center gap-3 rounded-lg border border-red-500/40 bg-red-950/80 px-4 py-2.5 text-sm text-red-200 shadow-lg backdrop-blur-md">
            <span>Session expired — please re-upload your document.</span>
            <button
              type="button"
              className="shrink-0 text-red-300 hover:text-white transition-colors"
              onClick={() => setSessionLost(false)}
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* LAYER 2: HTML UI Overlay */}
      <div className="pointer-events-none absolute inset-0 z-30 p-6">
        <div className="relative h-full w-full">
          {/* Top Header */}
          <div className="pointer-events-auto w-[min(360px,100%)]">
            <SpotlightCard className="rounded-xl">
              <Card className="bg-black/60 backdrop-blur-xl border-white/10">
                <CardHeader className="pb-2">
                  <CardTitle className="text-2xl font-bold text-white">Cognitive Cartographer</CardTitle>
                  <p className="text-sm text-muted-foreground">Rotate to explore the latent space.</p>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="text-xs text-muted-foreground">Nodes in scene: {nodes.length}</p>
                  <p className="mt-1 text-xs text-muted-foreground">Attention: {attentionStatus}</p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <p className="truncate text-[10px] text-muted-foreground/60">{apiHost}</p>
                    <button
                      type="button"
                      className="shrink-0 text-[10px] text-muted-foreground/60 underline hover:text-white/80 transition-colors"
                      onClick={() => {
                        localStorage.removeItem('cog-cart-api-host');
                        setApiHost('');
                        setApiHostInput('');
                      }}
                    >
                      change
                    </button>
                  </div>
                </CardContent>
              </Card>
            </SpotlightCard>
          </div>

          {/* Bottom-left Upload / AI Answer */}
          <div className="pointer-events-auto absolute bottom-0 left-0 w-[min(360px,100%)]">
            {queuePosition !== null ? (
              <Card className="bg-black/60 backdrop-blur-xl border-white/10">
                <CardContent className="p-3">
                  <div className="flex items-center gap-3">
                    <span className="relative flex size-2.5 shrink-0">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
                      <span className="relative inline-flex size-2.5 rounded-full bg-cyan-500" />
                    </span>
                    <p className="text-sm text-white/70">
                      Position <span className="font-semibold text-cyan-400">#{queuePosition}</span> in queue — waiting for the model...
                    </p>
                  </div>
                  <p className="mt-1.5 text-[11px] text-muted-foreground">The graph has loaded. You can explore it while waiting.</p>
                </CardContent>
              </Card>
            ) : aiAnswer !== null && !sessionLost ? (
              <Card className="flex h-[min(400px,calc(100vh-200px))] flex-col overflow-hidden bg-black/60 backdrop-blur-xl border-white/10">
                <CardHeader className="shrink-0 pb-0">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                      AI Answer
                    </CardTitle>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      className="text-white/70 hover:text-white"
                      onClick={() => {
                        setAiAnswer(null);
                        setAiSources([]);
                        setAiSourceNodes([]);
                        setActiveNodes([]);
                      }}
                    >
                      ✕
                    </Button>
                  </div>
                </CardHeader>
                <ScrollArea style={{ height: 'calc(100% - 24px)' }}>
                  <CardContent className="pb-4">
                    <TextGenerateEffect text={aiAnswer} />

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
                </ScrollArea>
              </Card>
            ) : (
              <Card className="bg-black/60 backdrop-blur-xl border-white/10">
                <CardContent className="p-3">
                  <FileUpload
                    className="p-4"
                    onChange={handleUploadFileSelection}
                    maxPdfSizeMb={maxPdfSizeMb}
                    onValidationError={(message) => {
                      setUploadErrorMessage(message);
                      if (message) {
                        setHasUploadedPdf(false);
                        setPendingUploadFile(null);
                        setUploadStatus(message);
                      }
                    }}
                  />
                  <p className={`mt-2 min-h-4.5 px-1 text-xs ${uploadErrorMessage ? 'text-red-400' : 'text-muted-foreground'}`}>
                    {uploadErrorMessage || (isUploading ? `Uploading: ${uploadStatus}` : uploadStatus)}
                  </p>
                  <p className="px-1 text-[11px] text-muted-foreground">
                    Server max PDF size: {maxPdfSizeMb > 0 ? `${formatMegabytes(maxPdfSizeMb)} MB` : 'No limit'}
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
            )}
          </div>

          {/* Bottom-Centered Load + Search */}
          <div className="pointer-events-none absolute bottom-0 left-1/2 -translate-x-1/2">
            <div className="pointer-events-auto flex flex-wrap items-end justify-center gap-3">
              <Card className="bg-black/60 backdrop-blur-xl border-white/10">
                <CardContent className="p-3">
                  <Button
                    variant={isLoading || isUploading || !sessionId ? 'secondary' : 'default'}
                    disabled={isLoading || isUploading || !sessionId}
                    onClick={handleLoadGraph}
                    title={!sessionId ? 'Upload a document first' : undefined}
                    className="bg-cyan-500 text-black hover:bg-cyan-400 disabled:bg-secondary disabled:text-muted-foreground"
                  >
                    {isLoading ? 'Loading...' : 'Load Graph'}
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-black/60 backdrop-blur-xl border-white/10">
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
            </div>
          </div>

          {/* Draggable Query / Model Card */}
          <div
            className="pointer-events-auto absolute bottom-0 right-0 z-40"
            style={{ transform: `translate(${queryCardOffset.x}px, ${queryCardOffset.y}px)` }}
          >
            <div
              className={`mb-2 flex touch-none select-none items-center justify-between rounded border border-white/10 bg-black/60 px-2 py-1 text-[10px] uppercase tracking-widest text-muted-foreground ${
                isQueryCardDragging ? 'cursor-grabbing' : 'cursor-grab'
              }`}
              onPointerDown={handleQueryCardDragStart}
              onPointerMove={handleQueryCardDragMove}
              onPointerUp={handleQueryCardDragEnd}
              onPointerCancel={handleQueryCardDragEnd}
            >
              <span>Query + Model</span>
              <span>Drag</span>
            </div>

            <GlowingBorder glowColor="rgba(168, 85, 247, 0.4)">
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
                      placeholder={sessionId ? 'Ask a question...' : 'Upload a document first'}
                      value={aiQuery}
                      disabled={!sessionId}
                      onChange={(e) => setAiQuery(e.target.value)}
                      className="w-72 bg-black/50 border-white/10 text-white placeholder:text-white/40 disabled:opacity-50"
                    />

                    <div className="flex items-center gap-2">
                      <span className="shrink-0 text-xs text-muted-foreground">model:</span>
                      <Select
                        value={selectedModelOption}
                        onValueChange={(val) => setSelectedModelOption(val as string)}
                        disabled={isLoadingModels || isSwitchingModel}
                      >
                        <SelectTrigger size="sm" className="w-52 bg-black/50 border-white/10 text-xs text-white">
                          <SelectValue placeholder="No models loaded" />
                        </SelectTrigger>
                        <SelectContent className="bg-[#1a1a2e] border-white/10">
                          {availableModels.map((modelName) => (
                            <SelectItem key={modelName} value={modelName}>
                              {modelName}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>

                      <Button
                        type="button"
                        size="xs"
                        variant="secondary"
                        disabled={isLoadingModels || isSwitchingModel}
                        onClick={() => void fetchAvailableModels()}
                      >
                        {isLoadingModels ? 'Loading...' : 'Refresh'}
                      </Button>
                    </div>

                    {allowCustomHfModels ? (
                      <>
                        <div className="flex items-center justify-between rounded border border-white/10 bg-black/40 px-2 py-1">
                          <span className="text-[11px] text-muted-foreground">Use custom HF model ID</span>
                          <Switch
                            size="sm"
                            checked={isCustomModelInputEnabled}
                            disabled={isLoadingModels || isSwitchingModel}
                            onCheckedChange={(checked) => setIsCustomModelInputEnabled(Boolean(checked))}
                            aria-label="Toggle custom HF model input"
                          />
                        </div>
                        <Input
                          placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct"
                          value={customModelName}
                          disabled={!isCustomModelInputEnabled || isSwitchingModel}
                          onChange={(e) => setCustomModelName(e.target.value)}
                          className="w-72 bg-black/50 border-white/10 text-white placeholder:text-white/40 disabled:opacity-55"
                        />
                        <span className="text-[11px] text-muted-foreground">
                          {isCustomModelInputEnabled
                            ? 'When non-empty, this custom ID overrides the dropdown model.'
                            : 'Using the selected dropdown model.'}
                        </span>
                      </>
                    ) : (
                      <span className="text-[11px] text-muted-foreground">
                        Custom HF models are disabled by the server. Choose one of the available models.
                      </span>
                    )}

                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        size="xs"
                        variant="secondary"
                        disabled={isSwitchingModel || pendingModelName.length === 0}
                        onClick={() => void handleChooseModel()}
                      >
                        {isSwitchingModel ? 'Switching...' : 'Use Model'}
                      </Button>
                      <span className="max-w-52 truncate text-xs text-muted-foreground">
                        {modelStatus || (activeModelName
                          ? `Using model: ${activeModelName}`
                          : availableModels.length > 0
                            ? 'No model selected'
                            : 'No models loaded')}
                      </span>
                    </div>

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
                    </div>

                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <span>Max prompt length:</span>
                      <span className="font-mono text-cyan-200/85">
                        {maxSeqLen > 0 ? `${maxSeqLen} tokens` : 'Server default'}
                      </span>
                    </div>
                  </div>
                  <Button
                    type="submit"
                    disabled={isQuerying || isSwitchingModel || !aiQuery.trim() || !sessionId}
                    title={!sessionId ? 'Upload a document first' : undefined}
                    className="self-start bg-purple-500 text-white hover:bg-purple-400 disabled:bg-secondary disabled:text-muted-foreground"
                  >
                    {isQuerying ? 'Thinking...' : 'Ask AI'}
                  </Button>
                </form>
              </CardContent>
            </GlowingBorder>
          </div>
        </div>
      </div>

      {/* LAYER 3: Attention Heatmap Popover */}
      {isAttentionPopoverOpen && (
        <div data-slot="attention-popover" className="pointer-events-auto fixed inset-0 z-40 bg-black/80 p-4 backdrop-blur-md">
          <div className="flex h-full w-full flex-col overflow-hidden rounded-xl border border-cyan-400/15 bg-[#0a0a18]">
            {/* Header bar */}
            <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
              <div className="flex items-center gap-4">
                <h2 className="text-sm font-semibold tracking-wide text-white">Attention Heads</h2>
                <Separator orientation="vertical" className="h-4 bg-white/15" />
                <div className="flex gap-3 text-[11px] text-muted-foreground">
                  <span>seq: {seqLen || '-'}</span>
                  <span>layers: {totalLayers || '-'}</span>
                  <span>heads: {totalHeads || '-'}</span>
                </div>
                <span className="text-[11px] text-cyan-300/70">{attentionStatus}</span>
              </div>
              <Button
                type="button"
                size="icon-xs"
                variant="ghost"
                className="text-white/60 hover:text-white"
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
                ✕
              </Button>
            </div>

            {/* Controls + Visualization */}
            <div className="flex min-h-0 flex-1">
              {/* Left sidebar controls */}
              {totalLayers > 0 && totalHeads > 0 && (
                <div className="flex w-48 shrink-0 flex-col gap-3 overflow-y-auto border-r border-white/10 bg-black/30 p-3">
                  {/* View mode tabs */}
                  <div>
                    <p className="mb-1.5 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">View</p>
                    <div className="flex flex-col gap-0.5">
                      {(['stack', 'layer', 'single'] as const).map((mode) => {
                        const label = mode === 'stack' ? '3D Stack' : mode === 'layer' ? 'Layer' : 'Single Head';
                        const disabled = mode === 'layer' ? totalLayers <= 0 : mode === 'single' ? (totalLayers <= 0 || totalHeads <= 0) : false;
                        const active = attentionViewMode === mode;
                        return (
                          <button
                            key={mode}
                            type="button"
                            disabled={disabled}
                            onClick={() => setAttentionViewMode(mode)}
                            className={`rounded px-2 py-1 text-left text-xs transition-colors ${
                              active
                                ? 'bg-white/10 text-white font-medium'
                                : 'text-muted-foreground hover:text-white hover:bg-white/5'
                            } disabled:pointer-events-none disabled:opacity-50`}
                          >
                            {label}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <Separator className="bg-white/10" />

                  {/* Layer / Head selectors */}
                  <div className="flex flex-col gap-2">
                    <div>
                      <p className="mb-1 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">Layer</p>
                      <Input
                        type="number"
                        min={1}
                        max={Math.max(totalLayers, 1)}
                        value={activeLayerIndex + 1}
                        onChange={(e) => setSelectedLayerIndex(Math.max(0, toPositiveInt(e.target.value, 1) - 1))}
                        className="h-7 bg-black/50 border-white/10 text-xs"
                      />
                    </div>
                    <div>
                      <p className="mb-1 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">Head</p>
                      <Input
                        type="number"
                        min={1}
                        max={Math.max(totalHeads, 1)}
                        value={activeHeadIndex + 1}
                        onChange={(e) => setSelectedHeadIndex(Math.max(0, toPositiveInt(e.target.value, 1) - 1))}
                        className="h-7 bg-black/50 border-white/10 text-xs"
                      />
                    </div>
                  </div>

                  <Separator className="bg-white/10" />

                  {/* Scale toggle */}
                  <div className="flex items-center gap-2">
                    <Switch
                      size="sm"
                      checked={useGlobalScale}
                      onCheckedChange={(checked) => setUseGlobalScale(Boolean(checked))}
                    />
                    <span className="text-[11px] text-muted-foreground">
                      {useGlobalScale ? 'Global scale' : 'Local scale'}
                    </span>
                  </div>

                  {/* Color legend */}
                  <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <span>0</span>
                    <div className="h-1.5 flex-1 rounded-full bg-linear-to-r from-black to-cyan-400" />
                    <span>{useGlobalScale ? '1.0' : 'max'}</span>
                  </div>

                  <Separator className="bg-white/10" />

                  {/* Token strips toggle */}
                  <Button
                    type="button"
                    size="xs"
                    variant="secondary"
                    className="w-full"
                    onClick={() => setShowAttentionMeta((prev) => !prev)}
                  >
                    {showAttentionMeta ? 'Hide tokens' : 'Show tokens'}
                  </Button>

                  <Separator className="bg-white/10" />

                  {/* Analyze heatmap */}
                  <div className="flex flex-col gap-1.5">
                    <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">Analyze</p>
                    <Textarea
                      value={analysisPrompt}
                      onChange={(e) => setAnalysisPrompt(e.target.value)}
                      rows={3}
                      className="min-h-0 max-h-20 overflow-y-auto bg-black/50 border-white/10 text-[11px] text-white placeholder:text-white/30 resize-none"
                      placeholder="Prompt for analysis..."
                    />
                    <Button
                      type="button"
                      size="xs"
                      variant="default"
                      className="w-full bg-cyan-600 text-white hover:bg-cyan-500"
                      disabled={isAnalyzing || isQuerying}
                      onClick={() => void handleAnalyzeHeatmap()}
                    >
                      {isAnalyzing ? 'Analyzing...' : 'Analyze View'}
                    </Button>
                  </div>
                </div>
              )}

              {/* Main visualization area */}
              <div className="flex min-h-0 flex-1 flex-col">
                {/* Token strips (collapsible) */}
                {showAttentionMeta && (
                  <div className="grid shrink-0 gap-2 border-b border-white/10 bg-black/20 p-3 xl:grid-cols-2">
                    <div>
                      <div className="mb-1 flex items-center gap-1">
                        <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                          Sequence ({promptTokens.length})
                        </p>
                        <span className="group relative inline-flex">
                          <span
                            aria-hidden="true"
                            className="flex size-3.5 cursor-help items-center justify-center rounded-full border border-white/20 text-[9px] font-semibold text-white/70"
                          >
                            ?
                          </span>
                          <span
                            role="tooltip"
                            className="pointer-events-none absolute left-0 top-full z-20 mt-1 w-56 rounded border border-white/15 bg-black/90 px-2 py-1 text-[10px] normal-case tracking-normal text-white/80 opacity-0 transition-opacity duration-150 group-hover:opacity-100"
                          >
                            Sequence tokens are the tokenized input context the model is currently attending over.
                          </span>
                        </span>
                      </div>
                      <div className="max-h-16 overflow-y-auto rounded border border-white/10 bg-black/40 p-1.5">
                        <div className="flex flex-wrap gap-1">
                          {promptTokens.length === 0 ? (
                            <p className="text-[11px] text-muted-foreground">Waiting...</p>
                          ) : (
                            promptTokens.map((token, idx) => (
                              <span
                                key={`${token}-${idx}`}
                                className="rounded border border-cyan-400/20 bg-cyan-500/10 px-1 py-px font-mono text-[10px] text-cyan-200"
                              >
                                {token}
                              </span>
                            ))
                          )}
                        </div>
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 flex items-center gap-1">
                        <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                          Generated ({generatedTokens.length})
                        </p>
                        <span className="group relative inline-flex">
                          <span
                            aria-hidden="true"
                            className="flex size-3.5 cursor-help items-center justify-center rounded-full border border-white/20 text-[9px] font-semibold text-white/70"
                          >
                            ?
                          </span>
                          <span
                            role="tooltip"
                            className="pointer-events-none absolute left-0 top-full z-20 mt-1 w-56 rounded border border-white/15 bg-black/90 px-2 py-1 text-[10px] normal-case tracking-normal text-white/80 opacity-0 transition-opacity duration-150 group-hover:opacity-100"
                          >
                            Generated tokens are the new output tokens produced step by step by the model.
                          </span>
                        </span>
                      </div>
                      <div className="max-h-16 overflow-y-auto rounded border border-white/10 bg-black/40 p-1.5">
                        {generatedTokens.length === 0 ? (
                          <p className="text-[11px] text-muted-foreground">Waiting...</p>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {generatedTokens.map((token, idx) => (
                              <span
                                key={`gen-${idx}`}
                                className="rounded border border-purple-400/20 bg-purple-500/10 px-1 py-px font-mono text-[10px] text-purple-200"
                              >
                                {token}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Heatmap canvas */}
                {totalLayers <= 0 || totalHeads <= 0 ? (
                  <div className="flex flex-1 items-center justify-center">
                    <p className="text-sm text-muted-foreground">Waiting for head-weight data...</p>
                  </div>
                ) : (
                  <div data-slot="attention-viz" className="min-h-0 flex-1">
                    {attentionViewMode === 'single' ? (
                      <SingleHeadView
                        heatmap={heatmapsRef.current[activeLayerIndex]?.[activeHeadIndex] ?? null}
                        layerIndex={activeLayerIndex}
                        headIndex={activeHeadIndex}
                        scaleMax={attentionScaleMax}
                        version={heatmapVersion}
                        tokens={promptTokens}
                      />
                    ) : attentionViewMode === 'layer' ? (
                      <LayerOverview
                        layerHeatmaps={heatmapsRef.current[activeLayerIndex]}
                        heads={totalHeads}
                        scaleMax={attentionScaleMax}
                        version={heatmapVersion}
                        selectedHeadIndex={activeHeadIndex}
                        onPickHead={(headIndex) => {
                          setSelectedHeadIndex(headIndex);
                          setAttentionViewMode('single');
                        }}
                        tokens={promptTokens}
                      />
                    ) : (
                      <AttentionOverview
                        heatmaps={heatmapsRef.current}
                        layers={totalLayers}
                        heads={totalHeads}
                        seqLen={Math.max(seqLen, seqLenRef.current)}
                        scaleMax={attentionScaleMax}
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

                {/* Analysis result */}
                {analysisResult !== null && (
                  <div className="shrink-0 border-t border-white/10 bg-black/30 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">Analysis</p>
                      <Button
                        type="button"
                        size="icon-xs"
                        variant="ghost"
                        className="text-white/60 hover:text-white"
                        onClick={() => setAnalysisResult(null)}
                      >
                        ✕
                      </Button>
                    </div>
                    <ScrollArea style={{ height: '120px' }}>
                      <div className="prose prose-invert prose-xs max-w-none pr-2 text-xs leading-relaxed text-white/85 [&_h1]:text-sm [&_h2]:text-xs [&_h3]:text-xs [&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-medium [&_p]:text-xs [&_li]:text-xs [&_ul]:pl-4 [&_ol]:pl-4 [&_code]:text-[10px] [&_code]:bg-white/10 [&_code]:px-1 [&_code]:rounded">
                        <Markdown>{analysisResult}</Markdown>
                      </div>
                    </ScrollArea>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* LAYER 4: Node Detail Side Panel */}
      {(() => {
        const isOpen = selectedNode !== null;
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
