import { create } from 'zustand';

export const generateRandomHex = () => {
  const randomColor = Math.floor(Math.random() * 16777215).toString(16);
  return "#" + randomColor.padStart(6, '0');
};

export interface GraphNode {
  id: string;
  label: string;
  text: string;
  x: number;
  y: number;
  z: number;
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface NodeData {
  id: string;
  label: string;
  text: string;
  position: [number, number, number];
  color: string;
}

export interface EdgeData {
  sourceId: string;
  targetId: string;
}

interface AppState {
  nodes: NodeData[];
  edges: EdgeData[];
  isLoading: boolean;
  activeNodeIds: Set<string>;
  aiSourceNodeIds: Set<string>;
  selectedNodeId: string | null;
  sessionId: string | null;
  uploadSocket: WebSocket | null;
  addNode: (label: string, position: [number, number, number], text?: string) => string;
  addEdge: (sourceId: string, targetId: string) => void;
  setLoading: (status: boolean) => void;
  setActiveNodes: (ids: string[]) => void;
  setAiSourceNodes: (ids: string[]) => void;
  setSelectedNode: (id: string | null) => void;
  loadGraph: (nodes: NodeData[], edges: EdgeData[]) => void;
  setSessionId: (id: string | null) => void;
  setUploadSocket: (ws: WebSocket | null) => void;
}



export const useAppStore = create<AppState>((set) => ({
  nodes: [],
  edges: [],
  isLoading: false,
  activeNodeIds: new Set<string>(),
  aiSourceNodeIds: new Set<string>(),
  sessionId: null,
  uploadSocket: null,

  addNode: (label, position, text = '') => {
    const id = Math.random().toString();
    set((state) => ({
      nodes: [...state.nodes, { id, label, text, position, color: generateRandomHex() }]
    }));
    return id;
  },

  addEdge: (sourceId, targetId) => set((state) => ({
    edges: [...state.edges, { sourceId, targetId }]
  })),
  selectedNodeId: null,
  setActiveNodes: (ids) => set((state) => ({
    activeNodeIds: new Set([...state.aiSourceNodeIds, ...ids])
  })),
  setAiSourceNodes: (ids) => set((state) => {
    const nextAiSourceNodeIds = new Set(ids);
    const nonSourceActiveIds = [...state.activeNodeIds].filter((id) => !state.aiSourceNodeIds.has(id));
    return {
      aiSourceNodeIds: nextAiSourceNodeIds,
      activeNodeIds: new Set([...nextAiSourceNodeIds, ...nonSourceActiveIds]),
    };
  }),
  setSelectedNode: (id) => set({ selectedNodeId: id }),
  loadGraph: (nodes, edges) => set({ nodes, edges, activeNodeIds: new Set(), aiSourceNodeIds: new Set(), selectedNodeId: null }),
  setLoading: (status) => set({ isLoading: status }),
  setSessionId: (id) => set({ sessionId: id }),
  setUploadSocket: (ws) => set({ uploadSocket: ws }),
}));
