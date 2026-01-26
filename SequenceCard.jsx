// src/components/SequenceCard.jsx
import React from "react";
import "../styles/sequencecard.css";

export default function SequenceCard({ card, index, onSelect }) {
  return (
    <div className="sequence-card" onClick={() => onSelect(card, index)}>
      <div className="sequence-card-header">
        <span className="sequence-card-index">#{index + 1}</span>
        <span className="sequence-card-type">{card.insight_type}</span>
      </div>

      {/* Thumbnail */}
      {card.thumbnail_path && (
        <img
          src={`http://localhost:8000${card.thumbnail_path}`}
          className="sequence-card-thumb"
          alt="thumbnail"
        />
      )}

      {/* Title */}
      <div className="sequence-card-measure">
        {card.measure} ({card.breakdown})
      </div>

      {/* Textual description */}
      <div className="sequence-card-description">
        {card.description || "No description available."}
      </div>
    </div>
  );
}
