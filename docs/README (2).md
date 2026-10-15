# Jarwis — AG Assistant

A context-aware personal operating assistant, implemented by the AG project.
Not a chatbot, not a smart speaker — a local-first executive layer that routes
between local and cloud reasoning, remembers what matters, and tracks its own
conversations and tasks.

Current version: **v0.29+ / AG Beta build-out**

## Assistant identity and personas

The assistant identifies as **Jarwis**. AG remains the repository and internal
project name. Jarwis has two presentation modes for the same assistant and
execution pipeline:

- **Jarwis** (default): composed, professional, observant, and subtly witty.
- **Ultron**: more assertive and sharply analytical, while retaining the same
  capabilities, safety checks, permissions, memory, and workspace.

Use `switch to Ultron mode`, `switch to Jarwis mode`, or `what mode are you in?`
in the CLI. The selected mode is saved in
`configuration/persona_config.json` and restored on the next launch.

---

## What AG actually is

AG is built in layers, each with one job:

```
USER
  ↓
Conversation Manager   (tracks topic/thread, resolves "it"/"that"/"all this")
  ↓
Working Memory         (session-only context: current topic, discussion buffer)
  ↓
Executive Layer        (decides what's needed before answering: Planning + Context Analysis)
  ↓
Cognitive Engine        [PLANNED]
  ↓
Memory / Tasks / Projects  ←→  Brain Selection (Local / Cloud / Auto)
  ↓
Mood & Delivery          [PLANNED]
  ↓
Final Response
```

The Cognitive Engine sits *before* the Brain in the target architecture:
Working Memory decides what AG needs to know, the Cognitive Engine (once
built) decides what kind of understanding should exist, and only then does
the Brain generate language.

---

## Build status

| Stage | Module | Status |
|---|---|---|
| 1 | Core (Brain Layer) | ✅ Complete |
| 2 | Memory | ✅ Complete |
| 3 | Project Context | ✅ Complete |
| 4 | Tasks | ✅ Complete |
| 5 | Executive Layer | 🟡 Partial (Storage/Category/Brain Routing/Planning/Context Analysis done) |
| 6 | Working Memory | ✅ Phase A complete (session context, discussion buffer, follow-up resolution, pipeline compatibility) |
| 7 | Cognitive Engine | 🟡 Phase A complete (context ingestion, intent classification, brain demand, scratchpad evaluation, `process_turn` pipeline entry point) |
| 8 | Mood Engine | ⬜ Planned |
| 9 | Interface (Matrix UI) | ⬜ Planned |
| 10 | Input (Voice + Vision) | ⬜ Planned |
| 11 | Automation | ⬜ Planned |

Plus, built alongside the core pipeline as the **AG Beta infrastructure layer**:

| Module | Status |
|---|---|
| Probe Simulator | ✅ Phase A — deterministic risk/outcome simulation before actions |
| Schema Processor | ✅ Phase A — normalizes/validates all AG document types |
| Storage Manager | ✅ Phase A — schema-aware JSON document persistence |
| Index Manager | ✅ Phase A — keyword/tag/entity search indexes |
| Control Layer | ✅ Phase A — Brain Registry + Authorization Manager |
| Conversation Manager | ✅ Phase A — topic tracking, reference resolution, pending state |
| AG Pipeline | ✅ Phase A — end-to-end cognitive pipeline orchestration |

**Build order:** finish Executive → Working Memory (expanded) → Cognitive Engine
(kept to a few concrete mechanisms, not a dashboard) → Mood → Interface/Input/Automation.

**AG Beta finish line:** Executive complete, Working Memory complete, Cognitive
Engine complete, Mood Engine complete, Matrix UI/Voice/Vision/Spatial complete.

**AG Alpha (after Beta):** no new subsystems — improve the brain itself
(reasoning, prediction, planning, learning, autonomy).

---

## Project layout

```
AG/
├── Jarwis.py                # Main loop: intent detection, routing, all user-facing behavior
├── brain.py                 # Local/cloud/auto brain selection and calling
├── local_brain.py           # Ollama (Qwen2.5:3B) local inference
├── executive.py             # Planning + Context Analysis (Executive Layer)
├── executive_layer.py       # Executive Layer Phase A — storage/category routing decisions
├── working_memory.py        # Session context, discussion buffer, follow-up resolution
├── cognitive_engine.py      # Prefrontal reasoning pipeline — intent classification, brain demand, process_turn
├── brain_dispatcher.py      # Brain adapter routing and dispatch
├── response_processor.py    # Response sanitization and safety layer
├── ag_pipeline.py           # End-to-end cognitive pipeline orchestration
├── conversation_manager.py  # Action pattern detection (brain switching) + conversation state
├── probe_simulator.py       # Risk/outcome simulation before actions execute
├── schema_processor.py      # Document normalization + validation (all AG document types)
├── storage_manager.py       # JSON document persistence (schema-aware, index-synced)
├── index_manager.py         # Keyword/tag/entity search indexes
├── control_layer.py         # Brain Registry + Authorization Manager
├── analytics_logger.py      # Append-only DWDM event log + basic stats
│
├── brain_config.json        # Persistent default brain mode (local/cloud/auto)
├── memory.json              # Long-term categorized memory
├── tasks.json                # Task list
├── project_context.json     # Project metadata/milestones
├── brain_registry.json      # Known brain providers/models for Control Layer
├── permissions.json         # Permission definitions for Control Layer
│
├── data/
│   ├── collections/         # Storage Manager's document collections
│   ├── indexes/             # Index Manager's search indexes
│   └── warehouse/events.jsonl  # DWDM event log
│
└── test_*.py                 # One standalone test file per Beta module
```

### Design principles this project holds itself to

- **One cohesive subsystem = one runtime file.** Split into a package only
  if a file exceeds ~700–1000 meaningful lines, migrations become
  substantial, or a single file becomes genuinely hard to maintain.
- **Incremental patches over full rewrites.** Changes are reviewed as
  diffs against the existing file, not regenerated from scratch.
- **Standard library only** for every Beta infrastructure module
  (Schema Processor, Storage/Index Manager, Control Layer, Conversation
  Manager, Probe Simulator) — no external dependencies, fully
  deterministic, no LLM calls inside any of them.
- **Layers don't duplicate responsibility.** Storage Manager owns
  persistence; Index Manager owns retrieval; Schema Processor owns
  validation; Control Layer owns permissions/brain selection — never
  both.

---

## Running AG

```bash
python Jarwis.py
```

Type naturally. A few structural commands:

| You say | AG does |
|---|---|
| `switch to cloud brain` | Permanently changes the default brain mode |
| `use cloud brain for briefing about all this` | Uses cloud for this one request only, then restores your default |
| `save last response` | Manually store the most recent brain answer |
| `shutdown` / `exit` / `bye` | Ends the session — if there's an unsaved discussion, AG asks once before storing it |

---

## Running tests

Each Beta module is tested standalone, no pytest required (plain asserts):

```bash
python test_working_memory.py
python test_cognitive_engine.py
python test_brain_dispatcher.py
python test_response_processor.py
python test_ag_pipeline.py
python test_executive_layer.py
python test_conversation_manager.py
python test_probe_simulator.py
python test_schema_processor.py
python test_storage_index.py
python test_control_layer.py
```

All tests run against temporary/isolated data — none of them touch your
real `memory.json`, `tasks.json`, or `data/` directory.

---

## Version history

```
9592f8a  Add Conversation Manager Phase A tests
5329bb4  AG Beta: Control Layer Phase A completed
191b990  AG Beta: consolidate Schema Processor into single module
771f3cd  AG Beta: Storage and Index Managers Phase A completed
ba645a7  AG Beta: Schema Processor Phase A completed
16c417f  feat(probe): implement Probe Simulator Phase A with deterministic risk analysis
811a2ec  AG v0.29 Working Memory Phase A conversation continuity foundation complete
fa9d6e0  AG v0.28 Executive Layer Phase B planning and context analysis complete
b90a033  AG v0.28 execution planning and context engine complete
f16b135  AG v0.28 Executive Layer Phase B planning and context analysis complete
3ea9feb  AG v0.27 Executive Layer Phase A complete
8df936d  AG v0.27 Executive Layer Phase A complete
d1190f7  AG v0.27 Executive Layer Phase A complete
84e4306  AG v0.26.4 brain phrase expansion complete
7e506be  AG v0.26.3 project phrase expansion complete
b620bbe  AG v0.26.1 memory phrase expansion complete
56a786a  AG v0.25 local brain operational
a61a1d0  AG v0.25 Offline brain operational
7722dd4  AG v0.22 brain and memory categories
74b8a56  AG v0.22 Phase A complete
9158b42  Initial AG prototype
```

---

## What's next

1. **Executive Layer Phase B completion** — close the remaining gap: wire
   `executive_layer.py` into `Jarwis.py`'s main loop so the Storage/Category
   decisions from `ExecutiveDecision` drive actual memory writes, replacing
   the older `executive.py` free functions.
2. **Cognitive Engine Phase B** — add concrete reasoning mechanisms:
   pattern recognition, causal chain evaluation, and self-verification
   against Working Memory facts. Keep it a small number of mechanisms,
   not a feature-label dashboard.
3. **Mood Engine** — tone/delivery layer that adjusts AG's response style
   based on cognitive mode and conversation state.
4. Eventually wire Conversation Manager, Working Memory, and Executive
   together as the single source of conversational truth, with Storage
   Manager / Index Manager / Schema Processor / Control Layer underneath
   as the persistence and safety layer.
