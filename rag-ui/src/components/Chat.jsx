import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || loading) return;

    const userMsg = { role: "user", content: input };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/ask/guarded`, {
        question: userMsg.content,
      });

      const sources = [
        ...new Set((res.data.sources || []).map((s) => s.source)),
      ];

      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.data.answer, sources },
      ]);
    } catch (err) {
      const status = err.response?.status;
      const content =
        status === 503
          ? "The safety check is unavailable. Try again in a moment."
          : status === 502
          ? "The model is unavailable right now. Try again."
          : "❌ Network error. Backend not reachable.";

      setMessages((m) => [...m, { role: "assistant", content }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-[65vh] rounded-xl border border-slate-800 bg-slate-900">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-sm text-slate-400">
            Ask a question about your uploaded documents…
          </p>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`max-w-[75%] rounded-lg px-4 py-2 text-sm ${
              m.role === "user"
                ? "ml-auto bg-blue-600 text-white"
                : "bg-slate-800 text-slate-100"
            }`}
          >
            {m.role === "assistant" ? (
              <ReactMarkdown>{m.content}</ReactMarkdown>
            ) : (
              m.content
            )}
            {m.sources?.length > 0 && (
              <p className="mt-2 text-xs text-slate-400">
                Sources: {m.sources.join(", ")}
              </p>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-slate-800 p-3 flex gap-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask something…"
          className="flex-1 rounded-md bg-slate-800 px-3 py-2 text-sm outline-none"
        />
        <button
          onClick={send}
          disabled={loading}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
        >
          {loading ? "Thinking…" : "Send"}
        </button>
      </div>
    </div>
  );
}
