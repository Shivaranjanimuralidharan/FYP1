// src/components/SuggestionPanel.jsx
import React from "react";
import "../styles/suggestion.css";

// src/components/SuggestionPanel.jsx

import SuggestionCard from "./SuggestionCard";


export default function SuggestionPanel({ suggestions = [], onSelect }) {
  // 🛑 HARD GUARD
  if (!Array.isArray(suggestions) || suggestions.length === 0) {
    return (
      <div className="suggestion-panel empty">
        <p>No alternative insights yet.</p>
      </div>
    );
  }

  return (
    <div className="suggestion-panel">
      {suggestions.map((item, idx) => {
        // 🛑 PER-ITEM GUARD
        if (!item || !item.insight) return null;

        const insight = item.insight;

        return (
          <div
            key={idx}
            className="suggestion-card"
            onClick={() => onSelect(insight)}
          >
            <div className="suggestion-card-header">
              <span className="insight-type">
                {insight.insight_type || "Insight"}
              </span>
              <span className="add-icon">＋</span>
            </div>

            {insight.thumbnail_path && (
              <img
                src={`http://localhost:8000${insight.thumbnail_path}`}
                alt="thumbnail"
                className="suggestion-thumb"
              />
            )}

            <div className="suggestion-desc">
              {insight.description || "Alternative insight"}
            </div>
          </div>
        );
      })}
    </div>
  );
}
