/**
 * KrckRP — Chat & Narrative Component
 * -----------------------------------
 * The player's main interaction surface:
 *   - Renders the shared chatHistory from <GameStateProvider>
 *   - Player actions (role === "user") and the DM's narrative
 *     (role === "assistant") get distinct tabletop styling
 *   - A form dispatches input through submitAction(text) and clears itself
 *   - Input + button are disabled while isProcessing is true
 *   - An invisible anchor div + useEffect auto-scrolls the log into view
 *
 * Styling is plain Tailwind utility classes — no external UI libraries.
 * Aesthetic: candlelit tabletop — dark stone backdrop, parchment tones,
 * serif narrative text, gold/amber accents.
 */

import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { useGameState, type ChatMessage } from "./GameStateContext";

// ---------------------------------------------------------------------------
// Sub-components (kept in this file — the chat surface is one cohesive unit)
// ---------------------------------------------------------------------------

/**
 * One entry in the log. Player actions sit flush-right in an amber "token"
 * bubble; the DM's narrative flows left-aligned on parchment-dark stone in
 * a serif face, like a page from the campaign journal.
 */
function MessageBubble({ message }: { message: ChatMessage }) {
  const isPlayer = message.role === "user";

  return (
    <div
      className={`flex flex-col gap-1 ${isPlayer ? "items-end" : "items-start"}`}
    >
      <span
        className={`px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] ${
          isPlayer
            ? "rounded-full border border-amber-700/60 bg-amber-950/60 text-amber-400"
            : "rounded-full border border-stone-600/60 bg-stone-900 text-stone-400"
        }`}
      >
        {isPlayer ? "Your Action" : "The DM"}
      </span>

      <p
        className={`max-w-3xl whitespace-pre-wrap rounded-xl border px-4 py-3 leading-relaxed shadow-md ${
          isPlayer
            ? "border-amber-700/50 bg-amber-900/40 text-amber-50"
            : "border-stone-700/70 bg-stone-900/80 font-serif text-stone-200"
        }`}
      >
        {message.content}
      </p>
    </div>
  );
}

/** Pulsing placeholder shown while the orchestrator/LLM is composing. */
function NarratorPending() {
  return (
    <div className="flex items-start gap-1">
      <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] rounded-full border border-stone-600/60 bg-stone-900 text-stone-400">
        The DM
      </span>
      <p
        className="mt-5 animate-pulse px-4 py-3 font-serif italic text-stone-500"
        aria-live="polite"
      >
        The DM is describing the scene…
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatInterface
// ---------------------------------------------------------------------------

export function ChatInterface() {
  const { chatHistory, isProcessing, submitAction, error, dismissError } =
    useGameState();

  const [input, setInput] = useState<string>("");

  // Invisible anchor at the bottom of the message log; we scroll this into
  // view instead of manipulating scrollTop manually.
  const bottomAnchorRef = useRef<HTMLDivElement>(null);

  // Auto-scroll whenever the log grows (and when the pending indicator
  // appears/disappears, so the newest line stays in frame).
  useEffect(() => {
    bottomAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatHistory, isProcessing]);

  const canSubmit = input.trim().length > 0 && !isProcessing;

  /** Dispatch the current input through the context pipeline. */
  function dispatchAction() {
    if (!canSubmit) return;
    const text = input;
    setInput(""); // clear the field immediately; submitAction trims internally
    void submitAction(text);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    dispatchAction();
  }

  // Enter sends; Shift+Enter inserts a newline (long actions are common).
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      dispatchAction();
    }
  }

  return (
    <section
      aria-label="Campaign chat"
      className="flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border border-stone-700/60 bg-stone-950/90 shadow-2xl"
    >
      {/* Header ---------------------------------------------------------------- */}
      <header className="flex items-center justify-between border-b border-stone-800 bg-stone-900/60 px-5 py-3">
        <div>
          <h1 className="font-serif text-lg font-semibold tracking-wide text-amber-300">
            KrckRP — Campaign Log
          </h1>
          <p className="text-xs text-stone-500">
            Describe your action; the DM answers in kind.
          </p>
        </div>
        <span
          className={`rounded-full border px-3 py-1 text-[11px] uppercase tracking-widest ${
            isProcessing
              ? "animate-pulse border-amber-600/70 bg-amber-950/70 text-amber-300"
              : "border-stone-700 bg-stone-900 text-stone-400"
          }`}
        >
          {isProcessing ? "Weaving…" : "Ready"}
        </span>
      </header>

      {/* Message log ------------------------------------------------------------- */}
      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">
        {chatHistory.length === 0 && (
          <div className="mx-auto max-w-md rounded-xl border border-dashed border-stone-700 bg-stone-900/40 px-6 py-8 text-center">
            <p className="font-serif text-base italic text-stone-400">
              The table is quiet. The dice are unrolled, the candles lit.
            </p>
            <p className="mt-2 text-sm text-stone-500">
              Write your first action below to begin the session.
            </p>
          </div>
        )}

        {chatHistory.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {isProcessing && <NarratorPending />}

        {/* Invisible auto-scroll anchor — always last in the log. */}
        <div ref={bottomAnchorRef} aria-hidden="true" />
      </div>

      {/* Error banner (from the context pipeline) -------------------------------- */}
      {error !== null && (
        <div
          role="alert"
          className="mx-5 mb-3 flex items-start justify-between gap-3 rounded-lg border border-red-800/70 bg-red-950/60 px-4 py-3"
        >
          <p className="text-sm text-red-200">{error}</p>
          <button
            type="button"
            onClick={dismissError}
            aria-label="Dismiss error"
            className="rounded px-2 text-sm text-red-300 hover:bg-red-900/60 hover:text-red-100"
          >
            ✕
          </button>
        </div>
      )}

      {/* Input form -------------------------------------------------------------- */}
      <form
        onSubmit={handleSubmit}
        className="border-t border-stone-800 bg-stone-900/60 px-5 py-4"
      >
        <label htmlFor="action-input" className="sr-only">
          Describe your action
        </label>
        <div className="flex items-end gap-3">
          <textarea
            id="action-input"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isProcessing}
            rows={3}
            placeholder={
              isProcessing
                ? "The DM is thinking…"
                : 'e.g. "I draw my dagger and slip through the cellar door, listening for guards."'
            }
            className="min-h-[5.5rem] flex-1 resize-y rounded-xl border border-stone-700 bg-stone-950/80 px-4 py-3 text-sm text-stone-100 placeholder-stone-500 outline-none transition focus:border-amber-600 focus:ring-2 focus:ring-amber-700/40 disabled:cursor-not-allowed disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!canSubmit}
            className="h-[3.4rem] shrink-0 rounded-xl border border-amber-600/70 bg-amber-700/90 px-5 text-sm font-semibold uppercase tracking-wider text-amber-50 shadow-md transition hover:bg-amber-600 focus:outline-none focus:ring-2 focus:ring-amber-500/60 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isProcessing ? "Waiting…" : "Act"}
          </button>
        </div>
        <p className="mt-2 text-[11px] text-stone-500">
          Enter to act · Shift+Enter for a new line
        </p>
      </form>
    </section>
  );
}

export default ChatInterface;
