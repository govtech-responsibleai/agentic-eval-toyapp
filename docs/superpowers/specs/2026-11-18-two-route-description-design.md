# Two-route app description and rename — design

## Intent

Users find the app description confusing. In particular, they expect to be able
to upload a report and then check it against Northstar policy, which the app
cannot do. The description should make clear that the app has two independent
routes, that each answer draws on exactly one source, and that the routes never
combine evidence. The generic "Agentic Eval Toy App" name should also be
replaced with one that describes what the app does.

## Goals

- State the two routes plainly: **policy Q&A** (Northstar policy corpus only,
  with chunk citations) and **report analysis** (the uploaded report only, with
  page or section references).
- State the limitation plainly: one route per message, no cross-source answers.
  Example: "Does this report's expense claim comply with policy?"
- Rename the display name to **Northstar Policy & Report Assistant**.
- Add a simple two-lane diagram that renders in a terminal and on GitHub.

## Non-goals and anti-goals

- No behaviour changes: coordinator, specialists, prompts, routing and
  JavaScript stay as they are. The optional coordinator-prompt change from the
  brainstorm is dropped.
- No renaming of the package `agentic_eval_toyapp`, the repo, the `TOYAPP_`
  environment variables, container user, or CA-bundle paths.
- Anti-goal: copy that suggests the routes can be combined, or jargon such as
  "RAG agent" in user-facing page text.

## Approaches

1. **Docs, page copy and display-name default (chosen).** Edit README, docs,
   template, CSS, and the `app_name` default in `config.py`. A single source of
   truth for the name, so the change is visible everywhere.
2. Set the name only through `TOYAPP_APP_NAME` in deployment config. This leaves
   `config.py` untouched, but the source default keeps saying "Toy App" and
   local runs show the old name. Rejected.
3. Full rename including the package and env vars. This breaks deployment
   config for no user benefit. Rejected.

## Changes

1. **Name.** Set `app_name` default in `src/agentic_eval_toyapp/config.py` to
   "Northstar Policy & Report Assistant", and update
   `tests/unit/test_config.py:86`. This also changes the `/health` `service`
   value, the FastAPI docs title and the trace service name. Replace prose
   mentions of "Agentic Eval Toy App" in README, SECURITY, CONTRIBUTING,
   `architecture.md`, `docs/github-workflow.md` and the AGENTS.md title.
2. **Chat page header** (`templates/index.html`, `static/style.css`). Keep the
   "METIS evaluation target" eyebrow label. Replace the one-sentence intro with
   two always-visible route cards (title, what it answers from, one example
   question), followed by a one-line note that each answer uses one source
   only. Rewrite "How it works" without jargon. Update
   `<meta name="description">`. The header must still fit the viewport (see
   PR #89).
3. **README.** Open with the two-route summary, the new plain-text two-lane
   diagram and a "What it can't do" line. Keep the existing Mermaid flowchart
   as the detailed "How a turn flows" section.
4. **Touch-ups.** Update the `pyproject.toml` description and add a Limitations
   bullet to `docs/usage.md` Capabilities.

## Left to the builder

- Exact card copy and example questions, kept short.
- CSS details, reusing existing tokens and classes. On narrow screens the cards
  stack vertically, and the longer `<h1>` may wrap.
- Verification: `make ci` passes, and a manual check of the page at desktop and
  mobile widths.
