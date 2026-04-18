declare module 'd3-force-3d' {
  export interface SimulationNode {
    index?: number;
    x?: number;
    y?: number;
    z?: number;
    vx?: number;
    vy?: number;
    vz?: number;
    fx?: number | null;
    fy?: number | null;
    fz?: number | null;
    [key: string]: any;
  }

  export interface SimulationLink<N extends SimulationNode = SimulationNode> {
    source: N | string | number;
    target: N | string | number;
    index?: number;
    [key: string]: any;
  }

  export interface Force<N extends SimulationNode = SimulationNode> {
    (alpha: number): void;
  }

  export interface Simulation<N extends SimulationNode = SimulationNode> {
    tick(iterations?: number): this;
    nodes(): N[];
    nodes(nodes: N[]): this;
    alpha(): number;
    alpha(alpha: number): this;
    alphaMin(): number;
    alphaMin(min: number): this;
    alphaDecay(): number;
    alphaDecay(decay: number): this;
    alphaTarget(): number;
    alphaTarget(target: number): this;
    velocityDecay(): number;
    velocityDecay(decay: number): this;
    force(name: string): Force<N> | undefined;
    force(name: string, force: Force<N> | null): this;
    find(x: number, y: number, z?: number, radius?: number): N | undefined;
    on(typenames: string): ((...args: any[]) => void) | undefined;
    on(typenames: string, listener: ((...args: any[]) => void) | null): this;
    stop(): this;
    restart(): this;
    numDimensions(): number;
    numDimensions(dims: number): this;
  }

  export function forceSimulation<N extends SimulationNode = SimulationNode>(nodes?: N[], numDimensions?: number): Simulation<N>;
  export function forceCenter<N extends SimulationNode = SimulationNode>(x?: number, y?: number, z?: number): Force<N> & { x(): number; x(x: number): any; y(): number; y(y: number): any; z(): number; z(z: number): any; strength(): number; strength(s: number): any };
  export function forceManyBody<N extends SimulationNode = SimulationNode>(): Force<N> & { strength(): number; strength(s: number | ((d: N, i: number, data: N[]) => number)): any; distanceMin(): number; distanceMin(d: number): any; distanceMax(): number; distanceMax(d: number): any };
  export function forceLink<N extends SimulationNode = SimulationNode, L extends SimulationLink<N> = SimulationLink<N>>(links?: L[]): Force<N> & { links(): L[]; links(links: L[]): any; id(): (node: N) => string | number; id(id: (node: N) => string | number): any; distance(): number | ((link: L, i: number, links: L[]) => number); distance(d: number | ((link: L, i: number, links: L[]) => number)): any; strength(): number | ((link: L, i: number, links: L[]) => number); strength(s: number | ((link: L, i: number, links: L[]) => number)): any };
  export function forceCollide<N extends SimulationNode = SimulationNode>(): Force<N> & { radius(): number; radius(r: number | ((node: N) => number)): any };
  export function forceX<N extends SimulationNode = SimulationNode>(x?: number): Force<N> & { strength(): number; strength(s: number): any; x(): number; x(x: number): any };
  export function forceY<N extends SimulationNode = SimulationNode>(y?: number): Force<N> & { strength(): number; strength(s: number): any; y(): number; y(y: number): any };
  export function forceZ<N extends SimulationNode = SimulationNode>(z?: number): Force<N> & { strength(): number; strength(s: number): any; z(): number; z(z: number): any };
  export function forceRadial<N extends SimulationNode = SimulationNode>(radius?: number, x?: number, y?: number, z?: number): Force<N> & { radius(): number; radius(r: number): any; strength(): number; strength(s: number): any; x(): number; x(x: number): any; y(): number; y(y: number): any; z(): number; z(z: number): any };
}
