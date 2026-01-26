// src/components/SequenceView.jsx
import React from "react";
import SequenceCard from "./SequenceCard.jsx";
import "../styles/sequenceview.css";


export default function SequenceView({ sequence, onSelectCard }) {
  return (
    <div className="sequence-view">
      <h2 className="sequence-title">Sequence View</h2>


      {sequence.length === 0 && (
        <div className="sequence-empty">No insights generated yet.</div>
      )}


      <div className="sequence-grid">
        {sequence.map((card, idx) => (
      <SequenceCard
        key={idx}
        card={card}
        index={idx}
        onSelect={onSelectCard}
      />
        ))}
      </div>
    </div>
  );
}
