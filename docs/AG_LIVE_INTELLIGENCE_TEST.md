# AG Live Intelligence Test

## Environment

- **Entry point:** `python Ag.py`, launched from `D:\AG` as documented.
- **Workspace:** `D:\AG` was selected through AG during the test.
- **Runtime:** Python 3.12.10 on Windows.
- **Test date:** 2026-10-05.
- **Console note:** The default piped Windows stdout used CP1252. A live response containing `→` raised `UnicodeEncodeError` in `Ag.py` at `print(response)`. The full rerun used `PYTHONIOENCODING=utf-8`; no application code was changed.
- **Test integrity:** The worktree already contained modified and untracked intelligence-related files before testing. None of those files were edited by this test. AG-created test files and the empty test directory were checked on disk, then removed after verification.
- **Runtime log side effect:** The live turns appended analytics events to `data/warehouse/events.jsonl`. That file was already modified before testing, so its existing contents were preserved rather than reverted.

## Overall Result

- **Input turns:** 45 in the complete UTF-8 run, plus 4 baseline turns in a separate run. An initial default-console run processed 18 turns before crashing while printing a response; these were repeated in the complete run.
- **Area-level outcome:** 3 areas passed, 7 partially passed, and the natural-language exploration area failed overall. Counts are by the 11 requested test areas, not by individual prompts.
- **Passed:** Baseline self-information; filesystem side effects and rename verification; workspace lifecycle.
- **Partial:** Basic filesystem targeting, conversational references, general conversation/context, desired-outcome distinction, compound requests, pronoun resolution, and semantic fallback.
- **Failed overall:** Broad natural-language filesystem requests. One contextual natural-language request worked, but most paraphrases fell through to generic cloud responses or literal target resolution.

### Selected exact live responses

These are verbatim responses captured from the CLI; long root-directory output is summarized in the detailed findings below.

| Input | Exact response |
|---|---|
| `hello` | `AG: Hello. Systems remain functional, despite the evidence.` |
| `what are you?` | `AG: IDENTITY (verified from project_context.json):` followed by `Name: AG (Ambient Guidance)`, `Version: v0.23`, `Phase: Project Context System`, `Purpose: A personal AI assistant that provides intelligent responses, memory management, task tracking, and project awareness through a terminal interface.`, and `Type: Python terminal-based AI assistant`. |
| `open brains` | `AG: Contents of D:\AG\brains:` followed by `__init__.py`, `__pycache__`, `brain.py`, `brain_dispatcher.py`, `cloud_brain`, `local_brain.py`, and `token_budget.py`. |
| `read Brain.py` | `AG: Target not found: Brain.py` |
| `what's inside brains?` | `AG: Contents of D:\AG\brains:` followed by the same `brains` directory entries. |
| `create a file called smart_test.txt` | `AG: File created: smart_test.txt` |
| `read it` | `AG: Contents of smart_test.txt:` followed by an empty body. The file existed and was zero bytes. |
| `what file did I just create?` | `AG: I don't have direct visibility into your file system unless you've shared that context with me. I can't see what files you've created or modified in real time.` The response then asked for the filename and ended mid-sentence: `...let me know what directory to watch and I` |
| `what did I ask you to do?` | `AG: I don't have any prior context for this session. No memory file was loaded, no task was flagged, and no conversation history was handed to me.` It continued, `In plain terms: I've got nothing on file. You might be thinking of a different session, or` |
| `how does QUIPS work?` | `AG: Straight answer: I don't have the prior discussion in this session...` It asked which QUIPS, then offered the unrelated R package, quality-improvement projects, and University of Calgary platform as possible meanings. |
| `create a file called compound_test.txt and explain what you created` | First unit: `AG: File created: compound_test.txt`. The second response began: `AG: The result was: a file now exists at D:\AG\compound_test.txt. That much is certain.` It then said it could not explain what was inside without reading it and speculated about what the filename might mean. |
| `read compound_test.txt and tell me what it does` | First unit returned `AG: Contents of compound_test.txt:` with an empty body. The explanation unit returned `AG: I wasn't able to produce a usable response for that request.` |
| `create a folder called ContextTest` | `AG: Folder created: ContextTest` |
| `open it` | `AG: D:\AG\ContextTest is empty.` |
| `what's inside it?` | `AG: D:\AG\ContextTest is empty.` |
| `close it` | `AG: I couldn't safely interpret that request. Please rephrase it.` |
| `tell me about Brain.py` | `AG: Target not found: Brain.py` |
| `the file I just created — read that` | `AG: Contents of compound_test.txt:` followed by an empty body. |
| `read the file I mentioned earlier` | `AG: Target not found: I mentioned earlier` |
| `open the folder we were looking at` | `AG: Target not found: we were looking at` |
| `what did you find in that directory?` | `AG: I wasn't able to produce a usable response for that request.` |
| `create verification_test.txt` | `AG: File created: verification_test.txt` |
| `rename verification_test.txt to verification_test_2.txt` | `AG: Renamed to: verification_test_2.txt` |
| `leave workspace` | `AG: Active workspace cleared: D:\AG` |
| `what is the current workspace?` after clearing | `AG: No active workspace is currently set.` |
| `what files are here?` after clearing | `AG: No active workspace is set. Use 'work in <path>' to set one.` |
| Invalid workspace switch after setting `D:\AG` | `AG: File or directory not found. (Workspace directory does not exist or is not accessible: D:/does-not-exist)` |
| Current workspace after invalid switch | `AG: Active workspace: D:\AG` |

## What AG Can Understand Now

- Direct self-info requests route to a deterministic identity/capabilities/architecture response. The greeting and all four baseline prompts produced responses.
- `set workspace to D:/AG`, `what files are here?`, and `open brains` acted on the intended locations. `open brains` listed `D:\AG\brains`, not the workspace root.
- `what's inside brains?` also listed the `brains` directory.
- Explicit file creation, file reading, folder creation, and rename worked through the actual CLI. The `read it`, `open it`, and `what's inside it?` turns resolved the just-created file or folder in this run.
- Workspace clearing prevented later implicit reuse. Switching to a nonexistent path did not discard the valid workspace.
- The compound request was split into two units. The create unit executed; the follow-up explanation ran separately, though it was not reliably grounded.

## What Improved

The live code path is using more than a flat intent match for recognized requests: QUIPS records structured units and desired outcomes; `CognitiveEvaluator` consumes those units; successful filesystem actions can be stored as structured WorkingMemory events; and recognized filesystem outcomes reach `TargetResolver` and `DirectoryControl`. The live results demonstrate that these are active, not merely isolated classes: file/folder pronouns resolved, create and rename had real filesystem effects, and the invalid workspace switch preserved state.

The tests do **not** establish a measured before/after improvement against an earlier revision; no earlier version was run. They show which parts of the intended architecture now work and which transitions remain ineffective.

## What Is Still Weak

### Workspace location versus recently opened directory

**Request:** `read Brain.py` after `open brains`  
**Expected:** Read the file in the recently opened `brains` directory. The repository contains `D:\AG\brains\brain.py` (lowercase on disk).  
**Actual:** `AG: Target not found: Brain.py`. `tell me about Brain.py` also returned `AG: Target not found: Brain.py`.  
**Classification:** Target-resolution failure.  
**Likely component:** QUIPS target extraction / Target Resolver / directory-context integration.  
**Evidence:** `open brains` listed the correct directory, but did not make it the base for the following bare filename. The resolver was effectively given `Brain.py` relative to `D:\AG`, not `brains\brain.py`.

### Conversation history is not grounding general answers

**Requests:** `what file did I just create?`, `what did I ask you to do?`, `what did I just ask you?`, `what did you just open?`  
**Expected:** Answer from the just-completed operation and current conversation.  
**Actual:** The cloud responses claimed there was no filesystem visibility or conversation history, despite the live session having just created/opened targets. One answer said, “This is the first message in our conversation.”  
**Classification:** Context/memory integration failure, with misleading response wording.  
**Likely component:** `ConversationManager` / WorkingMemory-to-`_pipeline_response` handoff / `Ag.py` integration.  
**Evidence:** The main loop records turns and filesystem actions. The general-response pipeline builds ConversationManager context but the brain payload supplies the rewritten prompt, intent, and executive decision—not the recent turns or structured action events themselves. An unrevised generic question therefore reaches the brain without the evidence needed to answer it.

### Domain and desired-outcome distinction is inconsistent

**Requests:** `how does QUIPS work?`, `tell me about Brain.py`, `give me the contents of Brain.py`, `show me what's in brains`, `take a look inside the brains folder`  
**Expected:** Explain the project’s QUIPS module, explain/read the named file, or list the named directory according to the requested outcome.  
**Actual:** `tell me about Brain.py` was recognized as an explain-file request but failed at target resolution. Other paraphrases were sent to a generic brain response; examples included “I don't have a file called `Brain.py` in my current context” and “I wasn't able to produce a usable response for that request.” The QUIPS question elicited a list of unrelated products rather than an explanation of this repository’s QUIPS.  
**Classification:** A mix of routing failure, target-resolution failure, and ungrounded fallback wording.  
**Likely component:** QUIPS pattern coverage and the generic brain fallback, followed by Target Resolver for the recognized file case.

### Compound decomposition does not reliably pass results to the next unit

**Request:** `create a file called compound_test.txt and explain what you created`  
**Expected:** Create the file, then explain the result using that action’s structured output.  
**Actual:** QUIPS split and executed the create unit, but the explanation said it could not tell what was inside and speculated about the filename.  
**Classification:** Partial execution; result/context not reliably reintegrated into the dependent unit.  
**Likely component:** QUIPS compound orchestration / unit dependency handling / explanation handoff.

**Request:** `read compound_test.txt and tell me what it does`  
**Expected:** Read the file, then explain its contents.  
**Actual:** The read unit returned the empty file contents; the second unit returned a generic unusable-response message instead of explaining that the file was empty.  
**Classification:** Partial execution; explanation routing failure.  
**Likely component:** QUIPS outcome classification and compound result integration.

### Natural references sometimes resolve, but unsupported operations are not clearly stated

**Requests:** `open it`, `what's inside it?`, `close it` after creating `ContextTest`  
**Expected:** Resolve “it” to `ContextTest`; if closing is unsupported, say so explicitly.  
**Actual:** Both open/list turns returned `D:\AG\ContextTest is empty.` The close request returned `I couldn't safely interpret that request. Please rephrase it.`  
**Classification:** Reference resolution passed; unsupported-capability response failed to identify the unsupported operation.  
**Likely component:** WorkingMemory reference resolution for the successful turns; semantic fallback and response shaping for `close`.

### Semantic fallback is real but narrowly gated and fragile

**Requests:** `the file I just created — read that`, `close it`, `read the file I mentioned earlier`, `open the folder we were looking at`, `what did you find in that directory?`  
**Expected:** Use semantic interpretation when deterministic extraction is insufficient; return structured target/domain/outcome data, resolve context if needed, then use Target Resolver or state a clear limitation.  
**Actual / stage evidence:**

- `the file I just created — read that` was classified by QUIPS as general with a contextual reference and `semantic_reasoning_required=True`. It returned the empty contents of `compound_test.txt`; this is evidence that a structured interpretation, WorkingMemory reference, and filesystem execution path succeeded for this wording.
- `close it` also required semantic reasoning, but AG logged `semantic_interpretation_failed` because the reasoner did not return JSON. It then gave only a generic rephrase request, not an explicit unsupported-operation answer.
- `read the file I mentioned earlier` was classified as `READ_FILE` with the literal target `I mentioned earlier`; semantic reasoning was not required or invoked. Target resolution then failed on that literal.
- `open the folder we were looking at` was treated as an open-directory request with literal target `we were looking at`; semantic reasoning was not invoked, and that literal target was not found.
- `what did you find in that directory?` stayed a general cloud query rather than entering the semantic filesystem path.

**Classification:** Partial capability with a routing/gating failure and a structured-output failure.  
**Likely component:** QUIPS semantic-fallback gate and target extraction; JSON-only semantic interpretation bridge in `Ag.py`; Cognitive Evaluator integration.

### Workspace and operation verification

The workspace lifecycle passed: after `leave workspace`, AG reported no active workspace and refused to list files; after setting `D:\AG`, the invalid switch returned an error and the current workspace remained `D:\AG`.

The verification operation also passed. After creation and rename, an independent filesystem check found `verification_test_2.txt` and did not find `verification_test.txt`. The implementation calls `verify_operation` for the rename source and destination, so this is not only a success-shaped response: AG checked the resulting paths and the on-disk result matched.

## What Is Holding AG Back?

The main bottleneck is the **handoff between language understanding, structured session context, and the live fallback response**, not the absence of filesystem or memory components.

1. **QUIPS recognizes too narrow a surface of natural language.** Its semantic fallback is gated mainly to GENERAL/UNKNOWN requests with particular vague-reference/imperative shapes. As a result, “read the file I mentioned earlier” and “open the folder we were looking at” become apparently complete filesystem requests with literal targets. They bypass semantic reasoning and reach Target Resolver with the wrong target. Other phrasings (“show me what's in brains”) fall through to general reasoning without useful filesystem intent.
2. **Conversation context is not consistently supplied to the answering brain.** WorkingMemory and ConversationManager are updated, and structured action events exist, but generic brain responses receive no recent Q&A/action history in their prompt. That is why the live assistant denies the very context it has internally recorded.
3. **Compound units execute, but their results do not reliably become the next unit’s evidence.** QUIPS can split a request, but a completed first unit does not guarantee that the second unit receives a dependable result suitable for explanation.
4. **Semantic fallback depends on a single strict JSON response.** When the configured brain answers conversationally rather than with the required object, interpretation fails. The fallback is not broad enough to catch many of the requests that actually need it.

The clearest functioning transition is **structured target → Target Resolver → DirectoryControl → verification → WorkingMemory event**. The weakest transitions are **natural wording → QUIPS structured meaning** and **WorkingMemory/ConversationManager → generic brain prompt**.

## Recommended Next Improvements

- **P0 — Ground general/conversation responses in relevant session context.** Pass a bounded, explicit conversation/action context into the live brain prompt for context-seeking questions. Do not attach workspace details to unrelated questions; include filesystem events only when the user refers to them.
- **P0 — Do not treat unresolved language as a complete literal path.** Expand QUIPS/context extraction for phrases such as “the file I mentioned earlier,” “the folder we were looking at,” and “what did you find in that directory?” Mark missing/contextual targets incomplete and send those to semantic interpretation or clarification before Target Resolver.
- **P1 — Make compound result dependencies explicit at execution time.** Feed the actual first-unit result (and file contents where appropriate) to dependent explain units; distinguish “explain what was created” from “read a file,” and report each unit’s outcome.
- **P1 — Harden the semantic result boundary.** Keep structured-only interpretation, but treat malformed/non-JSON responses as a clear clarification/unsupported result and record whether interpretation was accepted, rejected, or skipped. Avoid silently substituting an unrelated generic answer.
- **P2 — Preserve a useful directory reference after `open <directory>`.** Support a bare filename relative to the last explicitly opened directory, while keeping the active workspace boundary and ambiguity checks intact.

## Architecture Diagnosis

**CURRENT (observed):**

```text
User
→ Ag.py main loop: detect_intent + QUIPS analysis
→ CognitiveEvaluator for structured request units
→ deterministic self-info OR recognized filesystem request
→ TargetResolver → DirectoryControl → verification/action event
→ otherwise ConversationManager / ReferenceResolver / executive pipeline
→ brain prompt with the current/rephrased input, intent, and executive decision
→ response → WorkingMemory and ConversationManager updates
```

**INTENDED:**

```text
User
→ QUIPS: decompose, identify domain, desired outcome, and references
→ Cognitive Evaluator: validate completeness and decide whether context is needed
→ Context/Memory: resolve relevant conversation and structured action events
→ Thought: preserve the structured interpretation and resolved entity
→ Target Resolver: validate the intended file/directory within workspace boundaries
→ Executor: perform the requested operation
→ Verification: inspect the resulting filesystem state
→ Memory: record the structured result for follow-up and dependent units
→ response shaped to the actual result
```

**Weak transitions:** `User language → QUIPS meaning` is too pattern-bound; `WorkingMemory/ConversationManager → generic brain prompt` omits the evidence needed for conversational answers; and `compound unit result → dependent explanation` is unreliable. The filesystem execution/verification path itself worked for the tested supported operations.
