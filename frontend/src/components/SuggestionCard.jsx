import React from "react";
import "../styles/sequencecard.css";

export default function SuggestionCard({ insight, onAdd }) {
  return (
    <div className="sequence-card suggestion-card">
      <div className="sequence-card-header">
        <span className="sequence-card-type">
          {insight.insight_type}
        </span>

        <span
          className="sequence-card-add"
          onClick={(e) => {
            e.stopPropagation();
            onAdd(insight);
          }}
        >
          ➕
        </span>
      </div>

      {insight.thumbnail_path && (
        <img
          src={`http://localhost:8000${insight.thumbnail_path}`}
          className="sequence-card-thumb"
          alt="thumbnail"
        />
      )}

      <div className="sequence-card-measure">
        {insight.measure} ({insight.breakdown})
      </div>

      <div className="sequence-card-description">
        {insight.description}
      </div>
    </div>
  );
}