const BASE = "http://localhost:8000";

export async function uploadCSV(formData) {
  const res = await fetch(`${BASE}/api/preprocess`, {
    method: "POST",
    body: formData,
  });
  return await res.json();
}

export async function renderSequence(runId) {
  const res = await fetch(`${BASE}/api/render_sequence/${runId}`, {
    method: "POST",
  });
  return await res.json();
}

export async function fetchTopK(runId) {
  const res = await fetch(`${BASE}/api/download/${runId}/topk.json`);
  return await res.json();
}

export async function fetchCharts(runId) {
  const res = await fetch(`${BASE}/api/download/${runId}/descriptions.json`);
  return await res.json();
}
