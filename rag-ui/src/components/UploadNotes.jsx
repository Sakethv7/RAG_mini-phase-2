import { useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function UploadNotes() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [uploading, setUploading] = useState(false);

  async function upload() {
    if (!file) return;

    setStatus("Uploading…");
    setUploading(true);

    const form = new FormData();
    form.append("file", file);

    try {
      await axios.post(`${API_BASE}/upload`, form);
      setStatus("✅ Document indexed successfully");
    } catch (err) {
      const msg = err?.response?.data?.detail || "Upload failed";
      setStatus(`❌ ${msg}`);
    }
    setUploading(false);
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
      <h2 className="font-medium mb-3">Upload document</h2>

      <div className="flex items-center gap-3">
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          className="block w-full text-sm text-slate-300
                     file:mr-4 file:rounded-md file:border-0
                     file:bg-slate-800 file:px-4 file:py-2
                     file:text-slate-200 hover:file:bg-slate-700"
        />

        <button
          onClick={upload}
          disabled={uploading || !file}
          className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploading && (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/60 border-t-white" />
          )}
          {uploading ? "Uploading…" : "Upload"}
        </button>
      </div>

      <p className="mt-2 text-xs text-slate-400">
        Supported: PDF, TXT, MD
      </p>

      {status && (
        <p className="mt-2 text-sm text-slate-300">{status}</p>
      )}
    </div>
  );
}
