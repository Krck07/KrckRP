/**
 * KrckRP — Character Sidebar
 * ---------------------------
 * The player's at-a-glance "hard state", pulled live from <GameStateProvider>.
 *
 * The container (Sidebar) reads `activeState` via the useGameState() hook and
 * fans the four slices of state out to pure, presentational sub-components —
 * each is exported alongside the container so they can be reused or tested in
 * isolation:
 *
 *   Stats      — name, level, a colour-coded health bar (hp / max_hp),
 *                and the current location name directly below the stats
 *   Inventory  — item grid (name + quantity), or a stylized empty state
 *   Quests     — title / description / status; "active" vs "completed" styled
 *                apart (bright marker vs dimmed + strikethrough + checkmark)
 *   NpcsHere   — anyone currently in the area, name + disposition score
 *
 * Data comes exclusively from the GameState dict defined in ./api (the exact
 * shape the orchestrator echoes back as `current_state`). No other source of
 * state is read.
 *
 * Styling is plain Tailwind utility classes — the same candlelit-tabletop
 * look as ChatInterface: dark stone backdrop, parchment tones, serif display
 * text, gold/amber accents.
 */

import { type ReactNode } from "react";
import { useGameState } from "./GameStateContext";
import type {
  InventoryItem,
  Location,
  NPCHere,
  Player,
  Quest,
} from "./api";

// ---------------------------------------------------------------------------
// Shared bits
// ---------------------------------------------------------------------------

/**
 * Centered panel eyebrow with hairline rules on either side — the visual
 * "section header" of the sidebar. Renders as an <h2> for a sane outline.
 */
function PanelTitle({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-3">
      <span aria-hidden="true" className="h-px flex-1 bg-stone-700/70" />
      <h2 className="shrink-0 text-[11px] font-semibold uppercase tracking-[0.28em] text-amber-300/90">
        {children}
      </h2>
      <span aria-hidden="true" className="h-px flex-1 bg-stone-700/70" />
    </div>
  );
}

/** A short dashed placeholder card used by the empty states below. */
function EmptyNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-stone-700 bg-stone-900/40 px-4 py-5 text-center">
      <p className="font-serif italic text-sm text-stone-400">{children}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small pure helpers (kept local — they only reason about state already in
// hand, never about context)
// ---------------------------------------------------------------------------

/** Fraction of health remaining, clamped to [0, 1]. Guards a max_hp of 0. */
function healthFraction(hp: number, maxHp: number): number {
  if (maxHp <= 0) return 0;
  return Math.max(0, Math.min(1, hp / maxHp));
}

/** Intuitive bar colour: green → amber → red as health drops. */
function healthBarColor(fraction: number): string {
  if (fraction > 0.6) return "bg-emerald-500";
  if (fraction > 0.3) return "bg-amber-500";
  if (fraction > 0) return "bg-red-500";
  return "bg-stone-700";
}

/** Tone a disposition score reads as (positive = friendly, negative = hostile). */
function dispositionMeta(score: number): {
  label: string;
  text: string;
  chip: string;
} {
  if (score > 0) {
    return {
      label: "Friendly",
      text: "text-emerald-400/90",
      chip: "border-emerald-700/60 bg-emerald-950/50 text-emerald-300",
    };
  }
  if (score < 0) {
    return {
      label: "Hostile",
      text: "text-red-400/90",
      chip: "border-red-700/60 bg-red-950/50 text-red-300",
    };
  }
  return {
    label: "Neutral",
    text: "text-stone-400",
    chip: "border-stone-700 bg-stone-900 text-stone-300",
  };
}

/** True when a quest's status string marks it as done. */
function isCompleted(quest: Quest): boolean {
  return quest.status.trim().toLowerCase() === "completed";
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/**
 * Name, level, a colour-coded health bar, and the current location name
 * directly beneath the stats.
 */
export function Stats({
  player,
  location,
}: {
  player: Player;
  location: Location | null;
}) {
  const fraction = healthFraction(player.hp, player.max_hp);
  const percent = Math.round(fraction * 100);
  // Keep a visible sliver when wounded but alive; a fully-empty bar when dead.
  const barWidth = player.hp > 0 ? Math.max(3, percent) : 0;

  return (
    <section aria-label="Character stats" className="space-y-4">
      <PanelTitle>Character</PanelTitle>

      <div>
        {/* Name + level */}
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="truncate font-serif text-2xl font-semibold tracking-wide text-stone-50">
            {player.name}
          </h3>
          <span className="shrink-0 text-xs font-semibold uppercase tracking-[0.2em] text-amber-400">
            Level {player.level}
          </span>
        </div>

        {/* Health bar */}
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold uppercase tracking-[0.2em] text-stone-400">
              Health
            </span>
            <span className="tabular-nums text-stone-300">
              {player.hp} / {player.max_hp}
            </span>
          </div>
          <div
            role="progressbar"
            aria-label="Health"
            aria-valuenow={player.hp}
            aria-valuemin={0}
            aria-valuemax={player.max_hp}
            aria-valuetext={`${player.hp} of ${player.max_hp}`}
            className="mt-1.5 h-3 overflow-hidden rounded-full border border-stone-700/70 bg-stone-900"
          >
            <div
              className={`h-full rounded-full transition-all duration-500 ease-out ${healthBarColor(fraction)}`}
              style={{ width: `${barWidth}%` }}
            />
          </div>
        </div>

        {/* Current location — directly below the stats */}
        <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-stone-800 bg-stone-900/50 px-3 py-2.5">
          <span aria-hidden="true" className="mt-0.5 text-amber-500/80">
            ⚑
          </span>
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.25em] text-stone-500">
              Current Location
            </p>
            <p className="truncate font-serif text-sm text-stone-200">
              {location ? location.name : "Unknown territory"}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------

/**
 * A two-column grid of the player's items (name + quantity). Falls back to a
 * stylized "Inventory is empty" note when the array has no entries.
 */
export function Inventory({ items }: { items: InventoryItem[] }) {
  return (
    <section aria-label="Inventory" className="space-y-3">
      <PanelTitle>Inventory</PanelTitle>

      {items.length === 0 ? (
        <EmptyNote>
          Inventory is empty
          <span className="mt-1 block text-xs not-italic text-stone-500">
            Your satchel holds nothing but echoes.
          </span>
        </EmptyNote>
      ) : (
        <ul className="grid grid-cols-2 gap-2">
          {items.map((item) => (
            <li
              key={item.item_id}
              className="flex items-center justify-between gap-2 rounded-lg border border-stone-800 bg-stone-900/60 px-3 py-2"
            >
              <span className="min-w-0 truncate text-sm text-stone-200">
                {item.item_name}
              </span>
              <span className="shrink-0 tabular-nums text-xs font-semibold text-amber-300">
                ×{item.quantity}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Quests
// ---------------------------------------------------------------------------

/**
 * The player's quest log. Active quests read bright with a ◆ marker; completed
 * quests are dimmed, struck through, and checked off — so the two states are
 * unmistakable at a glance.
 */
export function Quests({ quests }: { quests: Quest[] }) {
  return (
    <section aria-label="Quests" className="space-y-3">
      <PanelTitle>Quests</PanelTitle>

      {quests.length === 0 ? (
        <EmptyNote>No quests yet. The map is unwritten.</EmptyNote>
      ) : (
        <ul className="space-y-2">
          {quests.map((quest) => {
            const completed = isCompleted(quest);
            return (
              <li
                key={quest.quest_id}
                className={
                  completed
                    ? "rounded-xl border border-stone-800 bg-stone-900/30 px-4 py-3 opacity-75"
                    : "rounded-xl border border-amber-800/40 bg-amber-950/20 px-4 py-3"
                }
              >
                <div className="flex items-start gap-2.5">
                  <span
                    aria-hidden="true"
                    className={
                      completed
                        ? "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-stone-600 text-[11px] text-stone-400"
                        : "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-amber-500/70 bg-amber-500/10 text-[11px] text-amber-300"
                    }
                  >
                    {completed ? "✓" : "◆"}
                  </span>

                  <div className="min-w-0 flex-1">
                    <p
                      className={
                        completed
                          ? "font-serif text-sm font-semibold leading-snug text-stone-500 line-through decoration-stone-600"
                          : "font-serif text-sm font-semibold leading-snug text-amber-100"
                      }
                    >
                      {quest.title}
                    </p>

                    <p
                      className={
                        completed
                          ? "mt-1 text-xs leading-relaxed text-stone-600"
                          : "mt-1 text-xs leading-relaxed text-stone-400"
                      }
                    >
                      {quest.description}
                    </p>

                    <p
                      className={
                        completed
                          ? "mt-2 text-[10px] font-semibold uppercase tracking-[0.25em] text-stone-600"
                          : "mt-2 text-[10px] font-semibold uppercase tracking-[0.25em] text-amber-500"
                      }
                    >
                      {completed ? "Completed" : (quest.status || "Active")}
                    </p>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// NPCs present
// ---------------------------------------------------------------------------

/**
 * Anyone currently sharing the player's location, with their disposition
 * score. The score chip is colour-coded (green / neutral / red) so the sign is
 * legible even without reading the number.
 */
export function NpcsHere({ npcs }: { npcs: NPCHere[] }) {
  return (
    <section aria-label="NPCs in the area" className="space-y-3">
      <PanelTitle>In the Area</PanelTitle>

      {npcs.length === 0 ? (
        <EmptyNote>No one else is here.</EmptyNote>
      ) : (
        <ul className="space-y-2">
          {npcs.map((npc) => {
            const meta = dispositionMeta(npc.disposition_score);
            return (
              <li
                key={npc.id}
                className="flex items-center justify-between gap-3 rounded-lg border border-stone-800 bg-stone-900/60 px-3 py-2.5"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-stone-100">
                    {npc.name}
                  </p>
                  <p className={`text-[11px] ${meta.text}`}>{meta.label}</p>
                </div>
                <span
                  className={`shrink-0 rounded-md border px-2 py-1 text-xs font-semibold tabular-nums ${meta.chip}`}
                >
                  {npc.disposition_score > 0
                    ? `+${npc.disposition_score}`
                    : npc.disposition_score}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Sidebar — the composed container
// ---------------------------------------------------------------------------

/**
 * The full character sidebar. Reads `activeState` from context; before the
 * first successful action it is null and an in-theme placeholder is shown.
 */
export function Sidebar() {
  const { activeState } = useGameState();

  const shell =
    "flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border border-stone-700/60 bg-stone-950/90 shadow-2xl";

  if (activeState === null) {
    return (
      <aside aria-label="Character sidebar" className={shell}>
        <header className="border-b border-stone-800 bg-stone-900/60 px-5 py-3">
          <h1 className="font-serif text-lg font-semibold tracking-wide text-amber-300">
            Character
          </h1>
          <p className="text-xs text-stone-500">The campaign ledger.</p>
        </header>
        <div className="flex flex-1 items-center justify-center px-6 py-10">
          <div className="max-w-xs text-center">
            <p className="font-serif text-base italic text-stone-400">
              No character yet.
            </p>
            <p className="mt-2 text-sm text-stone-500">
              Take your first action to reveal your stats, satchel, and quests.
            </p>
          </div>
        </div>
      </aside>
    );
  }

  const { player, location, inventory, quests, npcs_here } = activeState;

  return (
    <aside aria-label="Character sidebar" className={shell}>
      <header className="flex items-center justify-between border-b border-stone-800 bg-stone-900/60 px-5 py-3">
        <div>
          <h1 className="font-serif text-lg font-semibold tracking-wide text-amber-300">
            Character
          </h1>
          <p className="text-xs text-stone-500">
            Your hard state, kept in the ledger.
          </p>
        </div>
        <span className="rounded-full border border-stone-700 bg-stone-900 px-3 py-1 text-[11px] uppercase tracking-widest text-stone-400">
          {player.name}
        </span>
      </header>

      <div className="flex-1 space-y-6 overflow-y-auto px-5 py-5">
        <Stats player={player} location={location} />
        <Inventory items={inventory} />
        <Quests quests={quests} />
        <NpcsHere npcs={npcs_here} />
      </div>
    </aside>
  );
}

export default Sidebar;
