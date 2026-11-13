import { useEffect, useRef, useState } from "react";
import { Expand, RotateCcw } from "lucide-react";
import { xyzText } from "./types";
import type { Atom } from "./types";

interface ViewerInstance {
  addModel: (text: string, format: string) => unknown;
  setStyle: (sel: unknown, style: unknown) => void;
  addLine: (line: unknown) => void;
  addLabel: (text: string, style: unknown) => void;
  addVolumetricData: (data: string, format: string, spec: unknown) => void;
  zoomTo: () => void;
  zoom: (factor: number) => void;
  rotate: (angle: number, axis: string) => void;
  render: () => void;
  resize: () => void;
  clear: () => void;
  getView: () => number[];
  setView: (view: number[]) => void;
}
type MolModule = {
  createViewer: (el: HTMLElement, options: unknown) => ViewerInstance;
};
export default function Viewer({
  atoms,
  lattice,
  cube,
  kind = "density",
  iso = 0.04,
}: {
  atoms: Atom[];
  lattice?: number[][] | null;
  cube?: string | null;
  kind?: string;
  iso?: number;
}) {
  const host = useRef<HTMLDivElement>(null),
    view = useRef<ViewerInstance | null>(null),
    lastGeometry = useRef("");
  const [error, setError] = useState(""),
    [repeat, setRepeat] = useState(1),
    [reset, setReset] = useState(0),
    [ready, setReady] = useState(false);
  const geometryKey = JSON.stringify({ atoms, lattice, repeat, reset });
  useEffect(() => {
    let cancelled = false;
    let observer: ResizeObserver | undefined;
    let owned: ViewerInstance | null = null;
    async function initialize() {
      try {
        const imported = await import("3dmol");
        if (cancelled || !host.current) return;
        const mol = ((imported as unknown as { default?: MolModule }).default ??
          imported) as unknown as MolModule;
        host.current.replaceChildren();
        owned = mol.createViewer(host.current, {
          backgroundColor: "#101e2a",
          antialias: true,
        });
        view.current = owned;
        observer = new ResizeObserver(() => {
          owned?.resize();
          owned?.render();
        });
        observer.observe(host.current);
        setReady(true);
      } catch (e) {
        if (!cancelled)
          setError(
            `3D 표시를 시작할 수 없습니다. ${e instanceof Error ? e.message : String(e)}`,
          );
      }
    }
    void initialize();
    return () => {
      cancelled = true;
      observer?.disconnect();
      owned?.clear();
      if (view.current === owned) view.current = null;
    };
  }, []);
  useEffect(() => {
    const viewer = view.current;
    if (!ready || !viewer) return;
    try {
      const sameGeometry = lastGeometry.current === geometryKey;
      const saved = viewer.getView();
      viewer.clear();
      const parsed = JSON.parse(geometryKey) as {
        atoms: Atom[];
        lattice?: number[][] | null;
        repeat: number;
      };
      const lattice = parsed.lattice;
      let points = parsed.atoms;
      if (lattice && parsed.repeat > 1) {
        points = [];
        for (let i = 0; i < parsed.repeat; i++)
          for (let j = 0; j < parsed.repeat; j++)
            for (let k = 0; k < parsed.repeat; k++)
              for (const atom of parsed.atoms)
                points.push({
                  symbol: atom.symbol,
                  position: atom.position.map(
                    (n, c) =>
                      n +
                      i * lattice[0][c] +
                      j * lattice[1][c] +
                      k * lattice[2][c],
                  ),
                });
      }
      viewer.addModel(xyzText(points), "xyz");
      viewer.setStyle(
        {},
        {
          stick: { radius: 0.15, colorscheme: "Jmol" },
          sphere: { scale: 0.28, colorscheme: "Jmol" },
        },
      );
      if (lattice) {
        const vertex = (i: number, j: number, k: number) => ({
          x:
            parsed.repeat *
            (i * lattice[0][0] + j * lattice[1][0] + k * lattice[2][0]),
          y:
            parsed.repeat *
            (i * lattice[0][1] + j * lattice[1][1] + k * lattice[2][1]),
          z:
            parsed.repeat *
            (i * lattice[0][2] + j * lattice[1][2] + k * lattice[2][2]),
        });
        for (let i = 0; i < 2; i++)
          for (let j = 0; j < 2; j++) {
            viewer.addLine({
              start: vertex(0, i, j),
              end: vertex(1, i, j),
              color: "#65b9c9",
              linewidth: 1.5,
            });
            viewer.addLine({
              start: vertex(i, 0, j),
              end: vertex(i, 1, j),
              color: "#65b9c9",
              linewidth: 1.5,
            });
            viewer.addLine({
              start: vertex(i, j, 0),
              end: vertex(i, j, 1),
              color: "#65b9c9",
              linewidth: 1.5,
            });
          }
        ["a", "b", "c"].forEach((axis, i) =>
          viewer.addLabel(axis, {
            position: vertex(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0),
            fontSize: 13,
            fontColor: "#a8e6df",
            backgroundOpacity: 0,
          }),
        );
      }
      if (cube) {
        viewer.addVolumetricData(cube, "cube", {
          isoval: iso,
          color: "#5edac0",
          opacity: 0.68,
          smoothness: 2,
        });
        if (kind !== "density")
          viewer.addVolumetricData(cube, "cube", {
            isoval: -iso,
            color: "#a896f4",
            opacity: 0.68,
            smoothness: 2,
          });
      }
      if (sameGeometry) viewer.setView(saved);
      else {
        viewer.zoomTo();
        viewer.rotate(60, "y");
        viewer.rotate(18, "x");
        viewer.zoom(parsed.atoms.length <= 3 ? 1.8 : 1.3);
      }
      viewer.render();
      lastGeometry.current = geometryKey;
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [ready, geometryKey, cube, kind, iso]);
  return (
    <div className="viewer-shell">
      <div
        ref={host}
        className="molecule-viewer"
        aria-label="분자 및 결정 3D 구조"
      />
      <div className="viewer-axis">
        <span className="axis-z">z</span>
        <span className="axis-y">y</span>
        <span className="axis-x">x</span>
        <i />
      </div>
      <div className="viewer-controls">
        {lattice && (
          <select
            aria-label="단위격자 반복"
            value={repeat}
            onChange={(e) => setRepeat(Number(e.target.value))}
          >
            <option value={1}>1 × 1 × 1</option>
            <option value={2}>2 × 2 × 2</option>
            <option value={3}>3 × 3 × 3</option>
          </select>
        )}
        <button
          className="icon-button"
          title="시점 초기화"
          aria-label="시점 초기화"
          onClick={() => setReset((n) => n + 1)}
        >
          <RotateCcw size={16} />
        </button>
        <button
          className="icon-button"
          title="구조 확대"
          aria-label="구조 확대"
          onClick={() => {
            view.current?.zoomTo();
            view.current?.render();
          }}
        >
          <Expand size={16} />
        </button>
      </div>
      <div className="viewer-caption">드래그하여 회전 · 스크롤하여 확대</div>
      {error && (
        <div className="viewer-error" role="alert">
          {error}
        </div>
      )}
    </div>
  );
}
