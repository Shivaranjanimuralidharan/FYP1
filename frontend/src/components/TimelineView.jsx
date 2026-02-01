// src/components/TimelineView.jsx
import React, { useRef, useState, useEffect } from "react";
import Plot from "react-plotly.js";
import "../styles/timeline.css";

export default function TimelineView({
  timestamps,
  columns,
  data,
  subspaceEnabled,
  subspaceRange,
  onSubspaceSelected
}) {
  const containerRef = useRef(null);

  // Drag selection state
  const [dragStartX, setDragStartX] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [rectStyle, setRectStyle] = useState(null);

  // Convert pixel → timestamp
  function pixelToTime(xPixel) {
    const box = containerRef.current?.getBoundingClientRect();
    if (!box) return null;

    const pct = (xPixel - box.left) / box.width;
    if (pct < 0 || pct > 1) return null;

    const idx = Math.floor(pct * (timestamps.length - 1));
    return timestamps[idx];
  }

  // Start drag
  function handleMouseDown(e) {
    if (!subspaceEnabled) return;
    setDragging(true);
    setDragStartX(e.clientX);

    setRectStyle({
      left: e.clientX,
      width: 0
    });
  }

  // Update drag rectangle
  function handleMouseMove(e) {
    if (!dragging || !subspaceEnabled) return;

    const start = dragStartX;
    const curr = e.clientX;

    setRectStyle({
      left: Math.min(start, curr),
      width: Math.abs(curr - start)
    });
  }

  // End drag → convert selection to timestamps
  function handleMouseUp(e) {
    if (!dragging || !subspaceEnabled) return;

    setDragging(false);

    const t1 = pixelToTime(dragStartX);
    const t2 = pixelToTime(e.clientX);

    if (t1 && t2) {
      const start = t1 < t2 ? t1 : t2;
      const end = t1 < t2 ? t2 : t1;
      onSubspaceSelected({ start, end });
    }

    setRectStyle(null);
  }

  // Listen globally for mouseup
  useEffect(() => {
    function up(e) {
      if (dragging) handleMouseUp(e);
    }
    window.addEventListener("mouseup", up);
    window.addEventListener("mousemove", handleMouseMove);

    return () => {
      window.removeEventListener("mouseup", up);
      window.removeEventListener("mousemove", handleMouseMove);
    };
  });

  return (
    <div
      className="timeline-view"
      ref={containerRef}
      onMouseDown={handleMouseDown}
    >
      {columns.map((col) => (
        <div key={col} className="timeline-chart-container">
          <h4 style={{ marginBottom: 4 }}>{col}</h4>

          <Plot
            data={[
              {
                x: timestamps,
                y: data[col],
                mode: "lines",
                line: { color: "#1976d2" },
                name: col
              }
            ]}
            layout={{
              autosize: true,
              height: 140,
              margin: { l: 50, r: 20, t: 8, b: 30 },  // ⬅ more axis space
              xaxis: { title: "", tickfont: { size: 10 } },
              yaxis: { tickfont: { size: 10 } }
            }}


            config={{ displayModeBar: false }}
            useResizeHandler
            style={{ width: "100%" }}
          />

          {/* Final selected subspace highlight */}
          {subspaceRange && (
            <div
              className="timeline-selection-rect"
              style={{
                left: (() => {
                  const box = containerRef.current?.getBoundingClientRect();
                  if (!box) return 0;
                  const idx = timestamps.indexOf(subspaceRange.start);
                  const pct = idx / timestamps.length;
                  return box.width * pct;
                })(),
                width: (() => {
                  const box = containerRef.current?.getBoundingClientRect();
                  if (!box) return 0;
                  const i1 = timestamps.indexOf(subspaceRange.start);
                  const i2 = timestamps.indexOf(subspaceRange.end);
                  const pct = (i2 - i1) / timestamps.length;
                  return box.width * pct;
                })(),
                top: "40px",
                height: "180px"
              }}
            ></div>
          )}

          {/* Drag preview rectangle */}
          {rectStyle && (
            <div
              className="timeline-selection-rect"
              style={{
                left: rectStyle.left + "px",
                width: rectStyle.width + "px",
                top: "40px",
                height: "170px"
              }}
            ></div>
          )}
        </div>
      ))}
    </div>
  );
}