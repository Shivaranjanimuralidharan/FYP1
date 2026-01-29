// src/pages/Dashboard.jsx
import React, { useEffect, useState } from "react";
import TimelineView from "../components/TimelineView.jsx";
import DataOptions from "../components/DataOptions.jsx";
import InsightPanel from "../components/InsightPanel.jsx";
import SequenceView from "../components/SequenceView.jsx";
import SuggestionPanel from "../components/SuggestionPanel.jsx";

import "../styles/dashboard.css";

function getRunId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("run_id");
}

export default function Dashboard() {
  const runId = getRunId();

  const [timestamps, setTimestamps] = useState([]);
  const [columns, setColumns] = useState([]);
  const [rawData, setRawData] = useState({});
  const [error, setError] = useState("");

  // Panel switching
  const [showInsightPanel, setShowInsightPanel] = useState(false);

  // Sequence handling
  const [sequence, setSequence] = useState([]);
  const [selectedCard, setSelectedCard] = useState(null);

  // Insight panel states
  const [selectedMeasure, setSelectedMeasure] = useState("");
  const [insightType, setInsightType] = useState("trend");
  const [breakdown, setBreakdown] = useState("day");

  // Subspace selection
  const [subspaceEnabled, setSubspaceEnabled] = useState(true);
  const [subspaceRange, setSubspaceRange] = useState(null);

  // RL Suggestions placeholder
  const [suggestions, setSuggestions] = useState([]);

  //newOne
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("data");  // "data" → DataOptions, "insight" → InsightPanel


  // -----------------------------
  // Load timeline data (Module 1.2 output)
  // -----------------------------
  useEffect(() => {
    async function fetchTimeline() {
      try {
        const res = await fetch(`http://localhost:8000/api/data/${runId}`);
        if (!res.ok) throw new Error("Failed to load timeline");
        const json = await res.json();

        setTimestamps(json.timestamps);
        setColumns(json.columns);
        setRawData(json.data);
        setSelectedMeasure(json.columns[0]);
      } catch (err) {
        console.error(err);
        setError(err.message);
      }
    }
    fetchTimeline();
  }, [runId]);

  

  // -----------------------------
  // ✨ Click card → populate Insight Panel
  // -----------------------------
  async function handleSelectCard(card) {
  setSelectedCard(card);

  // Build minimal state vector (placeholder)
  // In practice this comes from backend encoding
  const stateVector = new Array(128).fill(0);

  try {
    const res = await fetch("http://localhost:8000/get_alternatives", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state_vector: stateVector,
        clicked_insight: card,
      }),
    });

    const json = await res.json();
    setSuggestions(json.alternatives || []);
  } catch (err) {
    console.error("Error loading alternatives", err);
    setSuggestions([]);
  }
}

  // -----------------------------
  // 🔧 Generate Current Insight
  // -----------------------------
  async function handleGenerateCurrent() {
    if (!selectedMeasure) return;

    const payload = {
      measure: selectedMeasure,
      type: insightType,
      breakdown,
      subspace: subspaceEnabled ? subspaceRange : null,
    };

    const res = await fetch(
      `http://localhost:8000/api/${runId}/sequence/generate_one`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    const card = await res.json();

    // Overwrite selected card
    if (selectedCard) {
      const idx = sequence.indexOf(selectedCard);
      const copy = [...sequence];
      copy[idx] = card;
      setSequence(copy);
    }
  }
  async function loadSequence() {
  try {
    const res = await fetch(`http://localhost:8000/api/${runId}/sequence/combined`);
    if (!res.ok) {
      console.warn("Sequence not ready yet...");
      return;
    }

    const data = await res.json();
    setSequence(data.sequence)

  } catch (err) {
    console.error("Error loading sequence:", err);
  }
  }
  
  function handleSubspaceChange({ startPct, endPct }) {
  if (!timestamps.length) return;

  const startIdx = Math.floor((startPct / 100) * (timestamps.length - 1));
  const endIdx = Math.floor((endPct / 100) * (timestamps.length - 1));

  setSubspaceRange({
    start: timestamps[Math.min(startIdx, endIdx)],
    end: timestamps[Math.max(startIdx, endIdx)],
    startPct,
    endPct,
  });
}

  // -----------------------------
  // 🔄 Generate Subsequent Insights
  // -----------------------------
  async function handleGenerateSubsequent() {
    const payload = {
      measure: selectedMeasure,
      type: insightType,
      breakdown,
      subspace: subspaceEnabled ? subspaceRange : null,
    };

    const res = await fetch(
      `http://localhost:8000/api/${runId}/sequence/generate_subsequent`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );

    const json = await res.json();
    setSequence(json.sequence);
  }
  async function handleStart() {
  setLoading(true);

  // 1️⃣ Ask backend to render sequence
  await fetch(`http://localhost:8000/api/render_sequence/${runId}`, {
    method: "POST",
  });

  // 2️⃣ Poll until combined sequence exists
  const poll = setInterval(async () => {
    try {
      const res = await fetch(
        `http://localhost:8000/api/${runId}/sequence/combined`
      );

      if (!res.ok) return; // keep polling

      const data = await res.json();

      // 3️⃣ NOW update UI
      clearInterval(poll);
      setSequence(data.sequence);

      setMode("insight");          // 🔑 move here
      setShowInsightPanel(true);   // 🔑 move here
      setLoading(false);           // 🔑 important
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 1200);
}


  

  return (
    <div className="dashboard-main">
      {error && <div className="error-box">{error}</div>}

      <div className="top-section">
        {/* LEFT: TIMELINE VIEW */}
        <div className="timeline-container">
          <TimelineView
            timestamps={timestamps}
            columns={columns}
            data={rawData}
            subspaceEnabled={subspaceEnabled}
            subspaceRange={subspaceRange}
            onSubspaceSelected={setSubspaceRange}
          />
        </div>

        {/* RIGHT: DATA OPTION PANEL → INSIGHT PANEL */}
        <div className="right-panel">
          {mode === "data" ? (
            <DataOptions measures={columns} onStart={handleStart} loading={loading} />  
          ) : (
            <>
            <InsightPanel
            measures={columns}
            selectedMeasure={selectedMeasure}
            setSelectedMeasure={setSelectedMeasure}
            insightType={insightType}
            setInsightType={setInsightType}
            breakdown={breakdown}
            setBreakdown={setBreakdown}
            timestamps={timestamps}
            subspaceRange={subspaceRange}
            setSubspaceRange={setSubspaceRange}
            onGenerateCurrent={handleGenerateCurrent}
            onGenerateSubsequent={handleGenerateSubsequent}
          />


              <SuggestionPanel
                suggestions={suggestions}
                onSelect={(newInsight) => {
                  // Replace selected card
                  const idx = sequence.indexOf(selectedCard);
                  if (idx !== -1) {
                    const updated = [...sequence];
                    updated[idx] = newInsight;
                    setSequence(updated);
                  }
                }}
              />

            </>
          )}
        </div>
      </div>

      {/* BOTTOM: SEQUENCE VIEW */}
      <div className="sequence-section">
        <SequenceView sequence={sequence} onSelectCard={handleSelectCard} />
      </div>
    </div>
  );
}
