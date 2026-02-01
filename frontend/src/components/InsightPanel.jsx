// src/components/InsightPanel.jsx
import React from "react";
import "../styles/insightpanel.css";

export default function InsightPanel({
  measures = [],
  selectedMeasure,
  setSelectedMeasure,
  insightType,
  setInsightType,
  breakdown,
  setBreakdown,
  timestamps = [],
  subspaceRange,
  setSubspaceRange,
  onGenerateCurrent,
  onGenerateSubsequent,
}) {
  // ---------- SAFETY GUARDS ----------
  if (!measures.length || !timestamps.length) {
    return (
      <div className="insight-panel">
        <h2 className="panel-title">Insight Panel</h2>
        <div style={{ opacity: 0.6 }}>Loading…</div>
      </div>
    );
  }

  const safeRange = subspaceRange ?? {
    start: 0,
    end: timestamps.length - 1,
  };

  // ---------- RENDER ----------
  return (
    <div className="insight-panel">
      <h2 className="panel-title">Insight Panel</h2>

      {/* SUBSPACE RANGE SLIDER */}
      <div className="insight-row">
        <label>Subspace Range</label>

        <div className="range-slider">
          <input
            type="range"
            min={0}
            max={timestamps.length - 1}
            value={safeRange.start}
            onChange={(e) =>
              setSubspaceRange({
                start: Math.min(Number(e.target.value), safeRange.end),
                end: safeRange.end,
              })
            }
          />

          <input
            type="range"
            min={0}
            max={timestamps.length - 1}
            value={safeRange.end}
            onChange={(e) =>
              setSubspaceRange({
                start: safeRange.start,
                end: Math.max(Number(e.target.value), safeRange.start),
              })
            }
          />
        </div>
      </div>

      <div className="subspace-display">
        {timestamps[safeRange.start]} → {timestamps[safeRange.end]}
      </div>

      <div className="insight-row insight-row-inline">
  <div className="insight-col">
    <label>Measure</label>
    <select
      value={selectedMeasure}
      onChange={(e) => setSelectedMeasure(e.target.value)}
      className="insight-dropdown"
    >
      {measures.map((m) => (
        <option key={m} value={m}>{m}</option>
      ))}
    </select>
  </div>

  <div className="insight-col">
    <label>Type</label>
    <select
      value={insightType}
      onChange={(e) => setInsightType(e.target.value)}
      className="insight-dropdown"
    >
      <option value="distribution">Distribution</option>
      <option value="extreme">Extreme</option>
      <option value="trend">Trend</option>
      <option value="correlation">Correlation</option>
      <option value="similarity">Similarity</option>
      <option value="outlier">Outlier</option>
      <option value="seasonality">Seasonality</option>
      <option value="autocorrelation">Autocorrelation</option>
    </select>
  </div>
</div>


      {/* BREAKDOWN */}
      <div className="insight-row">
        <label>Breakdown</label>
        <select
          value={breakdown}
          onChange={(e) => setBreakdown(e.target.value)}
          className="insight-dropdown"
        >
          <option value="year">Year</option>
          <option value="month">Month</option>
          <option value="day">Day</option>
        </select>
      </div>

      {/* ACTIONS */}
      <div className="insight-btn-row">
        <button className="btn-generate" onClick={onGenerateCurrent}>
          Generate Current
        </button>
        {/*<button className="btn-generate" onClick={onGenerateSubsequent}>
          Generate Subsequent
        </button>*/}
      </div>
    </div>
  );
}