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
        family:
          "-apple-system, BlinkMacSystemFont, Segoe UI, Noto Sans KR, sans-serif",
        color: "#5a656e",
        size: 12,
      },
      margin: { l: 66, r: 24, t: title ? 40 : 20, b: 48 },
      title: title
        ? { text: title, font: { size: 13, color: "#242b30" } }
        : undefined,
      xaxis: {
        title: { text: xTitle },
        gridcolor: "#e4e7e7",
        zerolinecolor: "#aeb7bc",
        tickfont: { size: 11 },
        automargin: true,
      },
      yaxis: {
        title: { text: yTitle },
        gridcolor: "#e4e7e7",
        zerolinecolor: "#aeb7bc",
        tickfont: { size: 11 },
        automargin: true,
      },
      showlegend: false,
      hovermode: "closest",
      hoverlabel: {
        bgcolor: "#fff",
        bordercolor: "#c9cfd3",
        font: { color: "#242b30" },
      },
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
      style={{ height, minHeight: height }}
      aria-label={title ?? `${xTitle} 대 ${yTitle} 그래프`}
    />
  );
}
