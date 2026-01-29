import React, { useState } from "react";
import "../index.css";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function handleFileChange(e) {
    const f = e.target.files && e.target.files[0];
    if (f) {
      if (!f.name.toLowerCase().endsWith(".csv")) {
        setError("Please upload a CSV file.");
      } else {
        setError("");
        setSelectedFile(f);
      }
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) {
      if (!f.name.toLowerCase().endsWith(".csv")) {
        setError("Please upload a CSV file.");
      } else {
        setError("");
        setSelectedFile(f);
      }
    }
  }
  function handleDragOver(e) { e.preventDefault(); }

async function submitFile() {
  if (!selectedFile) {
    setError("Please upload a CSV file first.");
    return;
  }
  setLoading(true);
  setError("");
  try {
    const form = new FormData();
    form.append("file", selectedFile);

    const res = await fetch("http://localhost:8000/api/preprocess", {
      method: "POST",
      body: form,
    });

    // log status and headers for debugging
    console.log("POST /api/preprocess status:", res.status, res.statusText);
    console.log("response headers:", Array.from(res.headers.entries()));

    // read body as text first (safer). We'll attempt JSON.parse afterwards.
    const text = await res.text();
    console.log("raw response body:", text);

    let data = null;
    if (text && text.trim().length > 0) {
      try {
        data = JSON.parse(text);
      } catch (err) {
        console.warn("response is not valid JSON:", err);
        // If response is printed as: "Detail: ..." or contains a run id, try to extract run_id with regex
        const m = text.match(/"run_id"\s*:\s*"([^"]+)"/) || text.match(/run_id[:=]\s*([A-Za-z0-9_-]+)/);
        if (m) data = { run_id: m[1] };
      }
    } else {
      console.warn("response body is empty string");
    }

    if (!res.ok) {
      // server returned non-200; include server text in thrown error
      throw new Error(`Server returned ${res.status}: ${text}`);
    }

    if (!data || !data.run_id) {
      // helpful error message: show what we got so you can paste it here
      throw new Error("No run_id found in response. Server returned: " + (text || "<empty body>"));
    }

    const runId = data.run_id;
    // Redirect (no react-router): go to dashboard with query param
    window.location.href = `/dashboard?run_id=${encodeURIComponent(runId)}`;

  } catch (err) {
    console.error("upload error:", err);
    // show a clearer message in UI
    setError("Error while uploading or processing file: " + (err.message || "unknown"));
    setLoading(false);
  }
}


  return (
    <div className="upload-container">
      <h1>AURALYTIX</h1>

      <div className="upload-subheading">
        <div className="dot" />
        Upload your data for generation
      </div>

      <div className="upload-box" onDrop={handleDrop} onDragOver={handleDragOver}>
        <div className="upload-inner">
          <svg width="36" height="39" viewBox="0 0 36 39" fill="none">
            <rect x="3" y="8" width="30" height="23" rx="2" fill="white" opacity="0.12" />
            <text x="50%" y="60%" textAnchor="middle" fill="black" fontSize="10" fontWeight="700"></text>
          </svg>
          <p className="welcome-text">
              Auralytix is here to help you!
          </p>

          <label className="upload-btn">
            <input
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            Click to upload
          </label>

          <div className="upload-helper">or drag a CSV file here</div>
          {selectedFile && <div className="selected-file">Selected: {selectedFile.name}</div>}
        </div>
      </div>

      {error && <div className="upload-error" role="alert">{error}</div>}

      <div style={{ marginTop: 12 }}>
        <button className="upload-btn" onClick={submitFile} disabled={loading}>
          {loading ? "Processing..." : "Start Preprocessing"}
        </button>
      </div>

      <div className="upload-note">
        Your data is processed on the server and you will be redirected to the Dashboard when ready.
      </div>
    </div>
  );
}
