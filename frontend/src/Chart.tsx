import { useEffect, useRef } from "react";
import type { Data, Layout, PlotMouseEvent } from "plotly.js";
import Plotly from "plotly.js-dist-min";

const emptyLayout: Partial<Layout> = {};

export default function Chart({
  data,
  title,
  xTitle,
  yTitle,
  height = 280,
  layout = emptyLayout,
  onPoint,
}: {
  data: Data[];
  title?: string;
  xTitle?: string;
  yTitle?: string;
  height?: number;
  layout?: Partial<Layout>;
  onPoint?: (index: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const common: Partial<Layout> = {
      autosize: true,
      height,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {
        family: "Inter, Pretendard, Arial, sans-serif",
        color: "#93a9b9",
        size: 11,
      },
      margin: { l: 66, r: 24, t: title ? 40 : 20, b: 48 },
      title: title
        ? { text: title, font: { size: 13, color: "#d7e4ec" } }
        : undefined,
      xaxis: {
        title: { text: xTitle },
        gridcolor: "#21313e",
        zerolinecolor: "#304453",
        automargin: true,
      },
      yaxis: {
        title: { text: yTitle },
        gridcolor: "#21313e",
        zerolinecolor: "#304453",
        automargin: true,
      },
      showlegend: false,
      hovermode: "closest",
      ...layout,
    };
    let disposed = false;
    Plotly.react(el, data, common, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    }).then(() => {
      if (disposed) return;
      const div = el as HTMLElement & {
        on?: (n: string, f: (e: PlotMouseEvent) => void) => void;
      };
      if (onPoint)
        div.on?.("plotly_click", (e) => {
          const point = e.points[0];
          if (point) onPoint(point.pointIndex);
        });
    });
    const observer = new ResizeObserver(() => {
      if (disposed || !el.isConnected || !el.clientWidth || !el.clientHeight)
        return;
      // Plotly defers resizing internally; navigating away can remove the
      // graph before that promise settles.
      void Promise.resolve(Plotly.Plots.resize(el)).catch(() => {});
    });
    observer.observe(el);
    return () => {
      disposed = true;
      observer.disconnect();
      Plotly.purge(el);
    };
  }, [data, title, xTitle, yTitle, height, layout, onPoint]);
  return (
    <div
      className="chart"
      ref={ref}
      aria-label={title ?? `${xTitle} 대 ${yTitle} 그래프`}
    />
  );
}
