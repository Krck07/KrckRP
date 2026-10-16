/**
 * KrckRP game state — React Context + Provider + custom hook.
 *
 * Owns the client-side game state:
 *   - currentPlayerId : which player is acting (default test id "kaelen")
 *   - chatHistory     : user/assistant message log
 *   - activeState     : the nested state dict last returned by the backend
 *   - isProcessing    : true while the LLM is generating a narrative
 *   - error           : last failure message (null when healthy)
 *
 * Submit pipeline (submitAction):
 *   1. optimistically append the user's message to chatHistory
 *   2. set isProcessing = true, clear any previous error
 *   3. POST via sendPlayerAction(playerId, text)
 *   4. append the LLM narrative, replace activeState with fresh backend data
 *   5. set isProcessing = false (finally — runs on success AND failure)
 *
 * On failure the user's message stays in chatHistory (they typed it) and the
 * error surfaces via `error` for a UI banner; no fake assistant message is
 * ever injected.
 */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  sendPlayerAction,
  type GameState,
} from "./api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  /** Client-generated id, stable for React keys. */
  id: string;
  role: "user" | "assistant";
  content: string;
}

export interface GameStateContextValue {
  /** Player whose state/actions we are driving. */
  currentPlayerId: string;
  /** Allow switching players (e.g. a later player-select screen). */
  setCurrentPlayerId: (playerId: string) => void;

  /** Full chat log, oldest-first. */
  chatHistory: ChatMessage[];

  /**
   * Nested hard state from the backend:
   * { player (incl. resource_name / current_resource / max_resource),
   *   location, npcs_here, inventory, abilities, quests }.
   * null until the first successful action.
   */
  activeState: GameState | null;

  /** True while the LLM is generating the narrative response. */
  isProcessing: boolean;

  /** Human-readable message of the last failed action, or null. */
  error: string | null;
  /** Clear the error banner. */
  dismissError: () => void;

  /**
   * Optimistic dispatcher: appends the user message, calls the orchestrator,
   * then appends the narrative + refreshes activeState. Safe to call from any
   * component; no-ops on empty input or while another action is in flight.
   */
  submitAction: (text: string) => Promise<void>;
}

// ---------------------------------------------------------------------------
// Defaults & helpers
// ---------------------------------------------------------------------------

/** Test player id seeded in game_state.db for development. */
const DEFAULT_PLAYER_ID = "kaelen";

let messageCounter = 0;

/** Unique-enough id for React keys (randomUUID when available). */
function createMessageId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  messageCounter += 1;
  return `msg-${Date.now()}-${messageCounter}`;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const GameStateContext = createContext<GameStateContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function GameStateProvider({ children }: { children: ReactNode }) {
  const [currentPlayerId, setCurrentPlayerId] = useState<string>(DEFAULT_PLAYER_ID);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [activeState, setActiveState] = useState<GameState | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const dismissError = useCallback(() => setError(null), []);

  const submitAction = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (trimmed.length === 0 || isProcessing) return;

      // 1. Optimistically append the user's input.
      const userMessage: ChatMessage = {
        id: createMessageId(),
        role: "user",
        content: trimmed,
      };
      setChatHistory((prev) => [...prev, userMessage]);

      // 2. Enter loading state, clear stale errors.
      setError(null);
      setIsProcessing(true);

      try {
        // 3. Call the orchestrator.
        const response = await sendPlayerAction(currentPlayerId, trimmed);

        // 4. Append the LLM narrative and refresh hard state.
        const assistantMessage: ChatMessage = {
          id: createMessageId(),
          role: "assistant",
          content: response.narrative_response,
        };
        setChatHistory((prev) => [...prev, assistantMessage]);
        setActiveState(response.current_state);
      } catch (err) {
        // Keep the user's message; surface the failure for a UI banner.
        setError(
          err instanceof Error
            ? err.message
            : "Unexpected error while contacting the orchestrator.",
        );
      } finally {
        // 5. Always release the loading state.
        setIsProcessing(false);
      }
    },
    [currentPlayerId, isProcessing],
  );

  const value = useMemo<GameStateContextValue>(
    () => ({
      currentPlayerId,
      setCurrentPlayerId,
      chatHistory,
      activeState,
      isProcessing,
      error,
      dismissError,
      submitAction,
    }),
    [
      currentPlayerId,
      chatHistory,
      activeState,
      isProcessing,
      error,
      dismissError,
      submitAction,
    ],
  );

  return (
    <GameStateContext.Provider value={value}>
      {children}
    </GameStateContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Custom hook
// ---------------------------------------------------------------------------

/**
 * Access the KrckRP game state and the submitAction dispatcher.
 * Must be called inside a <GameStateProvider>.
 */
export function useGameState(): GameStateContextValue {
  const context = useContext(GameStateContext);
  if (context === null) {
    throw new Error(
      "useGameState must be used within a <GameStateProvider>.",
    );
  }
  return context;
}
