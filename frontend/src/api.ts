/**
 * KrckRP API client
 * -----------------
 * Typed fetch wrapper around the FastAPI orchestrator (see InterfaceMap.txt).
 * Zero dependencies beyond the browser `fetch` API.
 *
 * Endpoints used:
 *   POST /api/action            -> PlayerActionResponse
 *   GET  /api/history/{playerId}-> ChatHistoryResponse (restore log on refresh)
 *   GET  /health                -> flat liveness dict (no LLM / heavy init on the server)
 */

const API_BASE_URL = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types — mirror the orchestrator's Pydantic models and StateManager schema
// ---------------------------------------------------------------------------

/**
 * Players table: {id, name, hp, max_hp, level, current_location_id,
 * resource_name, current_resource, max_resource}
 */
export interface Player {
  id: string;
  name: string;
  hp: number;
  max_hp: number;
  level: number;
  current_location_id: string | null;
  /** Label of the player's primary resource pool (e.g. "Stamina", "Mana", "Focus", "Tech-Charges"). */
  resource_name: string;
  /** Current value of the primary resource (server clamps to [0, max_resource]). */
  current_resource: number;
  /** Maximum value of the primary resource pool. */
  max_resource: number;
}

/** Locations table: {id, name, base_description} */
export interface Location {
  id: string;
  name: string;
  base_description: string;
}

/** NPCs present at the player's location: {id, name, disposition_score} */
export interface NPCHere {
  id: string;
  name: string;
  disposition_score: number;
}

/** Inventory entry: {item_id, item_name, quantity} */
export interface InventoryItem {
  item_id: string;
  item_name: string;
  quantity: number;
}

/** Quest row: {quest_id, title, description, status} */
export interface Quest {
  quest_id: string;
  title: string;
  description: string;
  status: string;
}

/** Abilities table: {ability_id, player_id, name, resource_cost} */
export interface Ability {
  ability_id: string;
  name: string;
  /** Primary resource spent when the ability is used. */
  resource_cost: number;
}

/**
 * The consolidated state dict returned by StateManager.get_player_state()
 * and echoed back (freshly re-read) as `current_state` in every response.
 */
export interface GameState {
  player: Player;
  location: Location | null;
  npcs_here: NPCHere[];
  inventory: InventoryItem[];
  abilities: Ability[];
  quests: Quest[];
}

/** One ChromaDB hit from LoreManager.search_lore(): {id, text, metadata, distance} */
export interface RetrievedLore {
  id: string;
  text: string;
  metadata: Record<string, string>;
  distance: number;
}

/** PlayerActionResponse (response_model of POST /api/action) */
export interface PlayerActionResponse {
  /** Cleaned narrative — <state_update> blocks already stripped server-side. */
  narrative_response: string;
  /** Lore snippets retrieved for this action (most similar first). */
  retrieved_lore: RetrievedLore[];
  /** Freshly re-read hard state (player, location, npcs_here, inventory, abilities, quests). */
  current_state: GameState;
}

/** One persisted chat turn from GET /api/history/{player_id}. */
export interface ChatHistoryMessage {
  /** Server-assigned ordering key (1-based index into the returned window). */
  id: number;
  role: "user" | "assistant";
  content: string;
}

/** GET /api/history/{player_id} response: last 20 turns, oldest-first. */
export interface ChatHistoryResponse {
  player_id: string;
  messages: ChatHistoryMessage[];
}

/** Flat liveness dict from GET /health. */
export interface HealthResponse {
  status: string;
  kobold_base_url: string;
  model: string;
  state_db: string;
  chroma_db: string;
  chroma_collection: string;
  embedding_model: string;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/**
 * Thrown for every failed request: network failures, non-2xx statuses,
 * and malformed bodies. `status` is present for HTTP-level failures.
 */
export class ApiError extends Error {
  readonly status?: number;
  readonly rawBody?: string;

  constructor(
    message: string,
    options: { status?: number; rawBody?: string } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.rawBody = options.rawBody;
  }
}

// ---------------------------------------------------------------------------
// Client functions
// ---------------------------------------------------------------------------

/**
 * Send a player action to the orchestrator.
 *
 * POST {API_BASE_URL}/api/action
 *   body: { player_id, action_text, n_lore_results }
 *
 * @returns the parsed PlayerActionResponse on success (200).
 * @throws  ApiError on network failure, non-2xx status (e.g. 404 unknown
 *          player, 502 LLM failure), or a malformed JSON body.
 */
export async function sendPlayerAction(
  playerId: string,
  actionText: string,
  nLoreResults: number = 3,
): Promise<PlayerActionResponse> {
  if (!playerId || playerId.trim().length === 0) {
    throw new ApiError("player_id must be a non-empty string.");
  }
  if (!actionText || actionText.trim().length === 0) {
    throw new ApiError("action_text must be a non-empty string.");
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        player_id: playerId,
        action_text: actionText,
        n_lore_results: nLoreResults,
      }),
    });
  } catch {
    // fetch rejects on network-level failure: orchestrator not running,
    // CORS blocked, or the browser is offline.
    throw new ApiError(
      "Could not reach the KrckRP orchestrator. " +
        `Is it running at ${API_BASE_URL}?`,
    );
  }

  if (!response.ok) {
    const rawBody = await response.text().catch(() => "");
    throw new ApiError(
      `Orchestrator responded with ${response.status} ${response.statusText}.`,
      { status: response.status, rawBody },
    );
  }

  try {
    return (await response.json()) as PlayerActionResponse;
  } catch {
    throw new ApiError(
      "Orchestrator returned a 200 but the body was not valid JSON.",
      { status: response.status },
    );
  }
}

/**
 * Fetch the player's persisted chat history (last 20 turns, oldest-first).
 *
 * GET {API_BASE_URL}/api/history/{playerId}
 *
 * Used on mount / player switch so the conversation survives a page refresh.
 *
 * @returns the parsed ChatHistoryResponse on success (200).
 * @throws  ApiError on network failure, non-2xx status (e.g. 404 unknown
 *          player), or a malformed JSON body.
 */
export async function getChatHistory(
  playerId: string,
): Promise<ChatHistoryResponse> {
  if (!playerId || playerId.trim().length === 0) {
    throw new ApiError("player_id must be a non-empty string.");
  }

  let response: Response;
  try {
    response = await fetch(
      `${API_BASE_URL}/api/history/${encodeURIComponent(playerId)}`,
    );
  } catch {
    throw new ApiError(
      "Could not reach the KrckRP orchestrator. " +
        `Is it running at ${API_BASE_URL}?`,
    );
  }

  if (!response.ok) {
    const rawBody = await response.text().catch(() => "");
    throw new ApiError(
      `Orchestrator responded with ${response.status} ${response.statusText}.`,
      { status: response.status, rawBody },
    );
  }

  try {
    return (await response.json()) as ChatHistoryResponse;
  } catch {
    throw new ApiError(
      "Orchestrator returned a 200 but the body was not valid JSON.",
      { status: response.status },
    );
  }
}

/**
 * Liveness probe (GET /health) — no LLM or heavy initialization on the server,
 * safe to poll from a UI status indicator.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/health`);
  } catch {
    throw new ApiError(
      "Could not reach the KrckRP orchestrator. " +
        `Is it running at ${API_BASE_URL}?`,
    );
  }

  if (!response.ok) {
    const rawBody = await response.text().catch(() => "");
    throw new ApiError(
      `Health check failed with ${response.status} ${response.statusText}.`,
      { status: response.status, rawBody },
    );
  }

  try {
    return (await response.json()) as HealthResponse;
  } catch {
    throw new ApiError("Health check body was not valid JSON.", {
      status: response.status,
    });
  }
}
