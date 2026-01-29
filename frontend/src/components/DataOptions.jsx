// src/components/DataOptions.jsx
import React, { useState } from "react";
import "../styles/dataoptions.css";

export default function DataOptions({ measures, onStart }) {
  const [selected, setSelected] = useState(measures);
  const [breakdown, setBreakdown] = useState("day");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");

  // Toggle a measure on/off
  function toggleMeasure(m) {
    if (selected.includes(m)) {
      setSelected(selected.filter((x) => x !== m));
    } else {
      setSelected([...selected, m]);
    }
  }

  // Select all or deselect all
  function toggleAll() {
    if (selected.length === measures.length) {
      setSelected([]);
    } else {
      setSelected(measures);
    }
  }

  function handleStart() {
    onStart({
      measures: selected,
      breakdown,
      start,
      end,
    });
  }

  return (
    <div className="data-options">
      <h2 style={{ marginBottom: 12 }}>Data Options</h2>

      {/* Time Range */}
      <div className="data-options-row">
        <label>Time Range</label>
        <input
          type="text"
          placeholder="Start (optional)"
          value={start}
          onChange={(e) => setStart(e.target.value)}
          style={{ padding: 6, borderRadius: 6, border: "1px solid #ccc" }}
        />
        <input
          type="text"
          placeholder="End (optional)"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
          style={{
            marginTop: 6,
            padding: 6,
            borderRadius: 6,
            border: "1px solid #ccc",
          }}
        />
      </div>

      {/* Breakdown dropdown */}
      <div className="data-options-row">
        <label>Breakdown</label>
        <select
          value={breakdown}
          onChange={(e) => setBreakdown(e.target.value)}
          style={{ padding: 6, borderRadius: 6, border: "1px solid #ccc" }}
        >
          <option value="year">Year</option>
          <option value="month">Month</option>
          <option value="day">Day</option>
        </select>
      </div>

      {/* Variable selection */}
      <div className="data-options-row">
        <label>
          Variables
          <span
            onClick={toggleAll}
            style={{
              float: "right",
              cursor: "pointer",
              color: "#1976d2",
              fontSize: 13,
            }}
          >
            {selected.length === measures.length ? "Deselect All" : "Select All"}
          </span>
        </label>

        <div className="measure-list">
          {measures.map((m) => (
            <label key={m} style={{ display: "flex", gap: 8 }}>
              <input
                type="checkbox"
                checked={selected.includes(m)}
                onChange={() => toggleMeasure(m)}
              />
              {m}
            </label>
          ))}
        </div>
      </div>

      {/* Start button */}
      <button className="start-btn" onClick={handleStart}>
        Start
      </button>
    </div>
  );
}
