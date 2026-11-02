# MONSTER AI - architecture

Large-move intelligence for NSE equities. Every trading morning it ranks the liquid names by the **calibrated
probability of an unusually large intraday range**, sends a Telegram report before the open, freezes the
predictions, scores them after the close and retrains under fixed gates.

**Shadow mode only.** The system never places, modifies or cancels an order and holds no broker credentials. Its
output is research. See [honesty rules](#honesty-rules).

Companion documents: [README](../README.md) (overview and quick start), [OPERATIONS](OPERATIONS.md) (runbook),
[deployment guide](../deployment/README.md), [DATA_SOURCES](DATA_SOURCES.md) (generated source catalogue) and
[BLUEPRINT_TRACEABILITY](BLUEPRINT_TRACEABILITY.md) (every requirement of the original design blueprint and where it lives).

## Contents

1. [System context](#system-context)
2. [Whole-code architecture](#whole-code-architecture)
3. [Data flow](#data-flow)
4. [Daily timeline](#daily-timeline)
5. [The scan, step by step](#the-scan-step-by-step)
6. [News intelligence and the optional AI services](#news-intelligence-and-the-optional-ai-services)
7. [Packages](#packages)
8. [Point-in-time correctness](#point-in-time-correctness)
9. [Models and governance](#models-and-governance)
10. [Honesty rules](#honesty-rules)
11. [Scheduling and self-healing](#scheduling-and-self-healing)
12. [Network layer: curl_cffi, bulk fetching, rate limits](#network-layer-curl_cffi-bulk-fetching-rate-limits)
13. [Exchange calendar for any month and year](#exchange-calendar-for-any-month-and-year)
14. [Time zones](#time-zones)
15. [Storage and bounded retention](#storage-and-bounded-retention)
16. [Logging and diagnostics](#logging-and-diagnostics)
17. [Degradation matrix](#degradation-matrix)
18. [External training and off-box storage](#external-training-and-off-box-storage)
19. [Deployment topology](#deployment-topology)
20. [Security](#security)
21. [Resource budget](#resource-budget)
22. [Design decisions](#design-decisions)
23. [Deviations from the blueprint](#deviations-from-the-blueprint)

## System context

```mermaid
flowchart LR
    SRC["Public data sources<br/>Yahoo, NSE, RSS, GDELT<br/>no paid keys required"] -->|"curl_cffi,<br/>rate limited"| MON
    subgraph SERVER["Your server: 1 GB RAM, native systemd, no containers"]
        direction TB
        MON["MONSTER AI<br/>daemon + job processes"] <--> DB[("SQLite WAL<br/>+ logs, backups, models")]
    end
    MON -->|"09:12 IST report,<br/>alerts"| TG["Telegram bot"] --> YOU(("You"))
    YOU -.->|"sudo monster-ai /<br/>SSH tunnel"| MON
    MON <-.->|"optional"| S3[("S3-compatible store<br/>backups, signed models")]
    MON -.->|"optional"| GROQ["Groq LLM<br/>news extraction"]
    MON -.->|"optional"| JEV["Jev (TypeSafe AI)<br/>news judgments"]
    GHA["GitHub Actions worker<br/>external training"] <-.-> S3
```

Everything the system knows lives in **one SQLite file**. There is no message bus, no container and no second process
holding state: a crash can only lose the work of the current job, and every job is safe to run again.

## Whole-code architecture

The code base is a layered modular monolith. Each layer uses only the layers below it (the arrows to the outside world are the
HTTPS calls), and every layer also uses the platform package `app/` (configuration, clock, logging). The module lists behind each
box are in the [package table](#packages) below.

```mermaid
flowchart TB
    subgraph L0["Entry points - start and run everything below"]
        UNITS["systemd units"]
        CLI["app.cli"]
        DASH["dashboard<br/>read-only"]
    end
    subgraph L1["automation/ - control plane, orchestrates the layers below"]
        SCHED["scheduler, schedule<br/>runner, jobspec"]
        JOBS["jobs, scan,<br/>bootstrap"]
        HEALTHM["health, retention"]
        CALM["calendar"]
        CTXM["context, intel,<br/>graph_learning"]
    end
    subgraph L2["Domain logic - reads and writes the data layer"]
        FEAT["features"]
        MOD["models"]
        RANK["ranking"]
        NEWS["news_intelligence"]
        GRAPH["event_graph"]
        BT["backtesting"]
        NOTIF["notifications"]
    end
    subgraph L3["Data layer"]
        ING["ingestion<br/>http (curl_cffi), yahoo, nse"]
        DBL["database<br/>store, backup, S3"]
    end
    subgraph L4["app/ - platform, used by every layer"]
        PLAT["config, clock, logging, locks, resources, errors, doctor"]
    end
    EXT["External: Yahoo, NSE, RSS, GDELT, Telegram, S3, Groq, Jev"]
    UNITS ~~~ SCHED
    SCHED ~~~ FEAT
    FEAT ~~~ ING
    ING ~~~ PLAT
    ING -.->|"HTTPS"| EXT
    NOTIF -.->|"HTTPS"| EXT
```

Dependency rules, checked against the import graph:

- `database` and `ranking` import nothing but `app`; `event_graph` imports only `database`; `notifications` imports only
  `ranking.types` besides `app`.
- `features`, `ingestion` and `backtesting` use `automation.calendar` as a shared service (it is the only `automation`
  module a lower layer imports; `models.external` imports `automation.runner` lazily to run the model-sync job).
- Only `automation` and `app.cli` see every package; nothing below them imports upwards.
- The scheduler daemon never imports pandas, numpy, LightGBM or `requests`: jobs run in child processes, so the
  long-lived process stays small.

The control plane in more detail:

```mermaid
flowchart TB
    CLID["app.cli daemon"] --> RD["run_daemon"]
    RD --> SCH["Scheduler: tick every few seconds"]
    SCH --> ENG["ScheduleEngine<br/>SCHEDULE table, job_runs ledger,<br/>calendar and IST clock"]
    ENG --> DUE["due jobs<br/>(run key, attempt, window)"]
    DUE --> GATE{"first runnable job:<br/>disk and RAM ok for a heavy job?"}
    GATE -- "no: skip it, try the next due job" --> DUE
    GATE -- yes --> LAUNCH["python -m app.cli run-job NAME<br/>(child process)"]
    LAUNCH --> EXE["runner.execute_job<br/>locks, log context, RSS watchdog, timeout"]
    EXE --> BODY["jobs.NAME(ctx), scan.run_scan,<br/>bootstrap"]
    BODY --> LED[("job_runs ledger, kv, alerts")]
    SCH --> HB["heartbeat: systemd watchdog,<br/>file, DB row"]
    SCH --> BOOTQ["bootstrap until<br/>bootstrap.state = done"]
```

## Data flow

```mermaid
flowchart TB
    subgraph IN["Ingestion: curl_cffi, validated, deadline-bounded"]
        Y["Yahoo<br/>daily bars, quotes"]
        A["NSE archives<br/>official bhavcopy, F&O"]
        J["NSE API<br/>pre-open, GIFT, filings"]
        R["RSS, GDELT,<br/>filings, calendar"]
    end
    Y --> V["validation: schema, ranges, dates, size"]
    A --> V
    J --> V
    R --> NI["news_intelligence<br/>dedup, entities, events, priced-in"]
    V --> DB[("SQLite - single source of truth")]
    NI --> DB
    DB --> EG["event_graph<br/>chains, sector sensitivities"]
    DB --> F["features: one code path for training AND live"]
    F --> Q{"data-quality gate<br/>ok, degraded or blocked"}
    Q --> M["models.inference<br/>champion or labelled baseline"]
    M --> K["ranking: eligibility, costs, tiers, abstention"]
    EG --> K
    NI --> K
    K --> S["freeze immutable snapshot + predictions"]
    S --> D["notifications: idempotent delivery ledger"]
    D --> T(["Telegram report 09:12 IST"])
    S --> E["backtesting.evaluate after the close<br/>official bhavcopy"]
    E --> O[("outcomes, rolling metrics")]
    O --> EV(["evening report"])
    O --> TR["weekly walk-forward training<br/>gates, champion / challenger"]
    TR --> M
```

## Daily timeline

All times are Asia/Kolkata. The authoritative table is `SCHEDULE` in [automation/schedule.py](../automation/schedule.py);
the daemon derives "what is due" from the `job_runs` ledger, never from a clock tick.

```mermaid
flowchart TB
    subgraph EARLY["Any day (night and early morning)"]
        direction TB
        N1["02:00 Sat: train"] ~~~ N2["04:00 Sun: refresh_universe"] ~~~ N3["05:30: refresh_calendar,<br/>model_sync"]
    end
    subgraph MORN["Trading morning"]
        direction TB
        M1["06:00 collect_global"] --> M2["06:30 collect_news"]
        M2 --> M3["07:50 health_check"]
        M3 --> M4["08:00 collect_india"]
        M4 --> M5["08:10 collect_news (late)"]
        M5 --> M6["08:45 prelim_scan"]
        M6 --> M7["09:00:30 refresh_preopen"]
        M7 --> M8["09:08:45 final_scan<br/>report by 09:12, hard deadline 09:14:30"]
    end
    subgraph EVE["After the close"]
        direction TB
        E1["17:45 ingest_eod"] --> E2["18:15 evaluate"]
        E2 --> E3["20:00 health_check"]
        E3 --> E4["23:30 backup"]
        E4 --> E5["23:50 housekeeping<br/>retention, official-bar upgrade"]
    end
    classDef critical fill:#fde2e0,stroke:#c0392b,color:#000
    class M6,M7,M8 critical
```

| Time | Job | Purpose |
|------|-----|---------|
| 06:00 | `collect_global` | global closes, live context, GIFT Nifty (retried until 08:30) |
| 06:30 / 08:10 | `collect_news` | RSS (conditional GET), GDELT, filings, board meetings, event extraction |
| 07:50 | `health_check` | operational alerts (also 20:00) |
| 08:00 | `collect_india` | previous session completeness: official file first, Yahoo as repair |
| 08:45 | `prelim_scan` | preliminary report (**critical**); first repairs the previous session when `collect_india` left it below 90 % coverage |
| 09:00:30 | `refresh_preopen` | NSE pre-open book (**critical**) |
| 09:08:45 | `final_scan` | final report with pre-open gaps, delivered by 09:12 (**critical**, hard deadline 09:14:30) |
| 17:45 | `ingest_eod` | official end-of-day files |
| 18:15 | `evaluate` | outcomes, metrics, evening report |
| 23:30 / 23:50 | `backup`, `housekeeping` | verified online backup (+ off-box copy); retention, official-bar upgrade, basis repair, DB maintenance |
| Sat 02:00 | `train` | walk-forward retraining (skipped in external mode) |
| Sun 04:00 | `refresh_universe` | index / F&O membership, history for new names, event-graph re-seeding and weekly re-estimation of its edge sensitivities |
| daily 05:30 | `refresh_calendar`, `model_sync` | holiday calendar; pull signed models in external mode |
| every 5 min | `monster-ai-health.timer` | independent health check (works even when the daemon is down) |

Critical jobs preempt non-critical ones. Non-trading days run only the "any day" jobs. Jobs marked `catch_up` still run
after their window (or on a later day) when the server was down.

## The scan, step by step

```mermaid
sequenceDiagram
    autonumber
    participant SCH as Scheduler
    participant RUN as final_scan job
    participant NSE as NSE API
    participant DB as SQLite
    participant TG as Telegram

    SCH->>RUN: start at 09:08:45
    RUN->>DB: trading day? snapshot frozen?
    alt a frozen snapshot exists
        RUN->>TG: re-send it, no recompute
    else first run of the day
        RUN->>NSE: GIFT + pre-open book
        NSE-->>RUN: prices, or nothing
        Note over RUN: data-quality gate (freshness, coverage)
        alt blocked
            RUN->>DB: alert scan_final_failed, no report
        else ok or degraded
            Note over RUN: features, champion model or baseline, tiers
            RUN->>DB: freeze snapshot + predictions (one transaction)
            RUN->>TG: deliver via ledger, deadline 09:14:30
        end
    end
```

Key properties: the snapshot is frozen **before** delivery, so a Telegram outage can never lose or change a report; a
retry re-sends the frozen report; a report that would arrive after the deadline is withheld and an alert is queued; the
pre-open wait is bounded by the same deadline, so a slow provider cannot push the report past its cutoff.

## News intelligence and the optional AI services

News is **evidence shown in the report, not a model input**: the LightGBM forecast of P(range > 3 %) is computed from prices,
F&O, global and calendar features only. The news layer decides which catalysts are listed for a name, how they are labelled
(event type, severity, rumour or denial, direction, source class) and which other stocks an event propagates to through the
event graph. Optional AI services sharpen that labelling; with no key set the deterministic rules do all of it and no request
is ever made.

```mermaid
flowchart TB
    RAW[("news_raw + filings<br/>unprocessed rows")] --> RULES["Rules, always<br/>language, routine and re-published filings dropped,<br/>entities, event type, rumour / denial, severity,<br/>direction (weak evidence capped)"]
    RULES --> WORTH{"an event type<br/>or a known company?"}
    WORTH -- "no: nothing is spent" --> DROP["skipped as irrelevant"]
    WORTH -- yes --> CTX["Context for the readers, always<br/>when, by whom, which company and sector<br/>(never prices or the rules' verdict)"]
    CTX --> GATE{"news about a listed company,<br/>not first-hand, a rumour<br/>or a deal that matters?"}
    GATE -- no --> GROQ["Groq LLM, optional<br/>structured extraction + verbatim quote"]
    GATE -- yes --> WEB["Groq web check, optional<br/>browser_search, validated brief"]
    WEB -->|"the brief joins the context; a sourced denial<br/>makes a refuted rumour, half the severity"| GROQ
    GROQ --> M1["merge: only what the quote proves<br/>(type, severity, horizon, time, directions)"]
    M1 --> JEV["Jev, optional<br/>five typed questions in one request"]
    JEV --> M2["merge, conservative: retype only when Jev is sure<br/>and the rules are not, add rumour / denial flags,<br/>blend severity, direction only when readings agree"]
    M2 --> EV[("events + evidence<br/>method: rules, rules+web, rules+llm, rules+llm+jev")]
    RULES -.->|"the fallback when an AI step has no answer"| EV
    EV --> USE["catalysts in the report, event-graph propagation,<br/>'already priced in' check"]
```

**What the rules decide first, before any AI is asked** (`news_intelligence/events.py`). Three rules came out of a live run on real
feeds and filings, in which 1 098 articles and 419 filings became 855 events:

- *A filing that only re-publishes an earlier one is dropped*: the newspaper copy of the results, the audio recording or the
  transcript of the call. The first-hand filing is the event; the copy only repeats its words. 31 of the 178 filings that became
  events were such copies (typed `earnings_result`, severity up to 0.6, and a third of Jev's calls). A routine notice (trading
  window, record date, schedule of a meet) that says nothing specific is dropped too.
- *A type resting on passing words in the body alone is a lead, not a headline*: nothing in the headline, no filing category behind
  it, and less than `WEAK_EVIDENCE_SCORE` (3.0) of rule weight, i.e. one or two rules. It keeps its type, which may still be right,
  but its severity is capped at `WEAK_EVIDENCE_SEVERITY` (0.35) and the event's rules say `weak_evidence`. About one article in
  seven was typed from a stray word: "escalation" in a power-demand story became a geopolitical conflict at severity 0.95, "divest"
  in a story about FII selling an acquisition at 0.9.
- *A company-type event whose text names no listed company gets the same cap* (`cap_unattributed`, rule `no_listed_company`):
  "GST collections jump 22.9%" is not a company's results, and without the cap it was the top catalyst for five of sixteen
  unrelated stocks.

Exchange filings, official statements and a company's own releases are never capped, and a verified AI answer can still raise a
severity the rules capped.

All three services (the extraction, the web check and Jev) run behind the same safety rails (`news_intelligence/remote.py`), each with
its own budget, circuit breaker, cooldown, cache and health row, so a slow, rate-limited or broken provider can only make the rules
do the work, never delay the morning schedule or lose an item, and a failing web check never touches the extraction:

```mermaid
sequenceDiagram
    autonumber
    participant P as news pipeline
    participant S as AI service (Groq LLM, web check or Jev)
    participant K as kv: cache, counters, cooldown
    participant H as source_health
    participant API as provider API

    P->>S: judge this headline and text, with its context
    S->>K: answered for this exact text and context recently? (14 days, the web check 1 day)
    alt cache hit
        K-->>P: result: no request, no budget used
    else miss
        S->>S: circuit closed, cooldown over, call and time budget left?
        alt refused
            S-->>P: nothing, the rules decide
        else admitted
            S->>API: POST with bearer key and timeout
            opt Groq: 429 on the first model
                S->>API: same request to the fallback model
            end
            alt valid answer
                API-->>S: JSON
                S->>S: strict validation (schema, verbatim quote, known entities, bounded brief)
                S->>K: cache the result, count calls and tokens
                S->>H: success and latency
                S-->>P: validated result
            else key or endpoint rejected
                S->>H: failure, the circuit opens for this run
                S-->>P: nothing
            else quota, Retry-After above 30 s
                S->>K: cooldown until then (at most 6 h, survives between runs)
                S->>H: failure
                S-->>P: nothing
            else web check: the model answered as a call to a tool that does not exist (HTTP 400 tool_use_failed)
                S->>S: read the brief from the error body when it has the shape of one, else retry once (never a back-off)
                S-->>P: the brief (unverified: it can inform but never discount an item) or nothing
            else web check: tool rejected or request above the plan's token limit
                S->>K: back off 6 h (survives between runs)
                S->>H: failure
                S-->>P: nothing, the item is judged without the web check
            end
        end
    end
```

| | Groq LLM (`llm_extraction`) | Web check (`llm_web`) | Jev (`jev_judgment`) |
|---|---|---|---|
| Model | `openai/gpt-oss-120b`, falling back to `openai/gpt-oss-20b` (Groq's production open-weight models: the larger for accuracy, the smaller as headroom because Groq's rate limits are per model) | the same models, when they offer the `browser_search` tool (the gpt-oss family only) | `jev-1.14.0`, pinned (the `jev-latest` alias moves without notice) |
| What it is | a text model: reads the article and returns structured JSON | Groq's built-in, server-side web search (powered by Exa) driven by a gpt-oss model. A request of its own, because Groq documents that browser search cannot be combined with structured outputs | a decision model: reads the article as a typed state and returns calibrated probabilities. It is not a price forecaster and not a chat model |
| Asked | event type, entities with a direction, severity, horizon, event time, a verbatim evidence quote | do independent outlets report the same facts, did anyone deny or correct them, was the event already public; up to three facts and the hosts relied on | event type (27-way choice), rumour, denial, severity (5 levels), direction for the company in focus |
| Trusted when | the evidence quote is found word for word in the text and every entity resolves to a known instrument; an unverified answer is capped at confidence 0.4 and cannot change the type, severity, horizon, time or any direction (it can only add companies the text really names) | every field is validated (allowed words, bounded integers, real booleans, link-free single-line facts, plausible host names); acted on alone only for a **sourced denial** (it names a reputable publisher of the source registry: an exchange, an official body, a wire or a business newspaper) whose facts are not confirmed | probability and confidence reach `JEV_MIN_CONFIDENCE` (0.5); a rumour or denial flag needs p >= 0.5 + `JEV_MIN_CONFIDENCE` / 2 |
| Never | invents evidence | raises a severity or a confidence, clears a flag, runs for an exchange filing, an official statement or a company's own release, for macro or sector news or a story about no listed company, or in a replay or backtest | clears a rumour flag the rules set, or flags an exchange filing or official statement as a rumour |
| Where it adds | news the rules cannot read: unusual wording, implied events, entities by description | the one thing the text cannot tell: is the rumour or the unverified deal true, denied or old | calibrated confidence on the event type, rumour and denial detection, a second opinion on severity and direction |
| Caps | per news job: `LLM_MAX_ARTICLES_PER_RUN` (60), `LLM_MAX_SECONDS_PER_RUN` (240 s), and a 14-day cache by text and context | per news job: `LLM_WEB_MAX_PER_RUN` (8) and per IST day `LLM_WEB_MAX_PER_DAY` (16), 30 % of the LLM time box, a 1-day cache; HTTP 413 or a rejected tool backs it off for 6 h | per news job: `JEV_MAX_ITEMS_PER_RUN` (200), 180 s, and a 14-day cache by text, context, company and source class |

Where both answer, the direction of the company in focus is the mean of the two readings when they agree and *unknown* (0) when they
point opposite ways; a direction offered for a rumour or a denial is halved, as the rules do. The company asked about is the one
the rules or the LLM already signed the event for (for an order, the winner and not the customer), else the most prominent one.

**What the services see** (`news_intelligence/context.py`): the headline, the whole stored text (up to 6 000 characters), and a
context block of the facts a human reader would use: when the item was published and when it is being checked (always in IST), the
source with its class (exchange filing, official statement, company, news media, social) and tier (wire, business media,
aggregator), the filing category, the issuer of a filing with its sector, the companies the entity matcher found in the text and,
when the web check ran, its brief. The LLM also gets the definition of every event type (the same definitions Jev's choice question
carries), so the rules, the LLM and Jev classify against one vocabulary. Never a key, a position, a prediction, a **price, a return or
the rules' own verdict**: the AI is meant to be an independent second reader whose answer is then compared with the rules', and a
price or a hindsight fact would let it explain a move instead of reading the news. Free text in the context (feed names, search
results) is single-line, length-capped and stripped of control characters and of the tags that structure the prompt.
The web check searches the public web; nothing of yours is sent with the request. Its time share is a hard bound: every request of
every service is given at most what is left of the run's wall-clock budget (and is not started when almost nothing is left), so a
slow provider cannot overrun it. The web check and the extraction share Groq's per-model token quota; when it is spent the
extraction hands over to its fallback model and the web check backs off. A search is by far the most expensive call (about
7 000 tokens, measured: the pages it reads count as input tokens, charged to the per-minute and to the per-day limit, so about 28
of them use up a free-tier model's 200 000 tokens a day), hence a daily cap besides the per-job one (`LLM_WEB_MAX_PER_DAY`, counted per
IST day in `kv`): retries and manual runs cannot add up to a day of tokens that the extraction needs.

How it is observed: `monster-ai ai-test` sends one made-up item through the real code (and makes one real web search) and prints
what came back or why not; `monster-ai doctor` lists the models each key can use (no tokens spent); the counters `llm.*` /
`llmweb.*` / `jev.*` (calls, failures, tokens, cooldown) are in `kv` and in `monster-ai diag`; every service has a row in
`source_health`, so the health check warns after repeated failures; the report's coverage note names a degraded extraction or Jev;
every job's counts (used, cached, failed, skipped by budget, retyped, rumours, denials, web checks and refutations; summed over
the batches of a drained backlog) are in `news.last_run`, and what the web check found is kept in the event's evidence.

## Packages

| Package | Responsibility |
|---------|----------------|
| [app/](../app) | configuration (single typed catalogue in `config.py`, generates `.env.example`), injectable clock (UTC instants, IST market logic, display zone), logging (daily files, secret redaction, crash hooks) with the `logs` / `diag` tools, code fingerprint (`release.py`), file locks, resource guards (RSS watchdog, disk, RAM), CLI, `doctor`, error hierarchy |
| [ingestion/](../ingestion) | the curl_cffi HTTP transport (`http.py`: browser impersonation, adaptive per-source rate limit, retries, circuit breaker, deadlines), Yahoo (chart + bulk `spark` probe), NSE archive and API clients, validation, universe building, freshness / data-quality gate, split-adjusted reads (`adjust.py`) and basis-break repair (`basis.py`), news collectors (RSS conditional GET), source catalogue (`registry.py` -> [DATA_SOURCES.md](DATA_SOURCES.md)) |
| [database/](../database) | SQLite connection (WAL, busy timeout), versioned migrations, the `Store` repository, verified online backups with a size budget (optionally gpg-encrypted), S3-compatible object store (SigV4, stdlib only), off-box backup helpers |
| [features/](../features) | the single feature builder used by training and live scoring: technical, F&O, global, regime, calendar, sector, similar-day; label construction; schema hash and `FEATURE_VERSION`; the point-in-time panel loader |
| [news_intelligence/](../news_intelligence), [event_graph/](../event_graph) | deduplication, entity linking, event extraction (rules first, then the optional Groq LLM `llm.py`, its web check `web.py` and Jev `jev.py` behind the shared safety rails of `remote.py`; `context.py` holds the facts both readers receive; `ai_check.py` is the live self-check behind `ai-test`), source quality and provenance (rumour, source class, independent sources), "priced in" detection; supply-chain / sector impact propagation over a seeded expert-prior graph whose edges are re-measured weekly from returns ([automation/graph_learning.py](../automation/graph_learning.py): lagged regression, out-of-sample lag choice, sign stability, Benjamini-Hochberg control; edges without a price series stay labelled priors) with an append-only edge history |
| [models/](../models) | LightGBM specialists + meta model, isotonic calibration, walk-forward training, gates, artifact format (text models + JSON, **never pickle**), registry (champion/challenger, rollback), HMAC-signed bundles for external training |
| [ranking/](../ranking) | eligibility, conviction tiers, contradiction checks, cost estimates, abstention, baselines |
| [notifications/](../notifications) | Telegram client, report renderers (4096-char safe; `style.py` holds the emoji, bars and next-step hints, `samples.py` the made-up data behind `telegram-preview`), idempotent delivery with a ledger and deadline guard |
| [automation/](../automation) | the scan, all jobs, the schedule engine, the daemon, the job runner, bootstrap, health, the exchange calendar, bounded-storage retention |
| [backtesting/](../backtesting) | outcome evaluation (provisional -> official), metrics with day-block bootstrap intervals, the live range-forecast record and its slices, walk-forward backtest |
| [dashboard/](../dashboard) | optional read-only FastAPI dashboard (loopback by default) |
| [deployment/](../deployment) | installer, systemd units, remote deploy scripts - see [deployment/README.md](../deployment/README.md) |
| [scripts/](../scripts) | `generate_docs.py`: regenerates `.env.example` and `docs/DATA_SOURCES.md` from the code (tests fail when they drift) |

## Point-in-time correctness

A scan at time *T* may use only what was knowable at *T*.

- Price, volume and F&O features are **shifted by one session** (a morning scan never sees today's bar).
- Global series are used at each series' own availability: a US close is known in the morning, a live quote is
  taken at the scan cutoff, and a stale series is blanked (after 4 days; 7 for European and Asian equity series) instead
  of being forward-filled silently.
- Calendar and results-calendar features are known in advance and are not shifted.
- Stage `final` adds `p_*` gap features computed from the pre-open price only; names without a pre-open price fall
  back to the `pre` model (reported as such). Pre-open rows are stored only when stamped on the trade date (IST): outside
  the pre-open session NSE serves the *previous* session's book, and that must never become today's opening gap.
- **One price basis.** Official end-of-day rows are stored raw and re-based on the post-split basis when they are read
  (`ingestion.adjust.adjusted_bars`, used by features, graph learning and the news reaction check), so the order in which
  rows and split events arrive never matters. A split of a stock that already has history is detected from the exchange's
  own previous close (`ingestion.basis`): the vendor is asked for exactly that stock until it reports the split, and the
  operator is told after six sessions. On the ex-date morning the pre-open book exposes the break before the vendor does,
  and such a name is scored **without** its (meaningless) opening gap.
- **Point-in-time membership and survivorship.** F&O membership is a point-in-time flag; a training panel also holds the
  quarantined, suspended or delisted names that have bars in the window. Names the database never ingested stay unknown
  (documented limitation).
- Relative-volume baselines exclude the current session; similar-day features use the same fixed pool and window in
  training and live, so a live row equals the batch row for the same day.
- News and event features are computed from `first_seen_at` (when the system saw an item), never from later edits;
  event-graph edges keep an append-only history so replays can use the edges as they were (opt-in).
- Muhurat and other special sessions are removed from the training panel (market-wide volume far below the
  trailing median) and never count as a "missing" session.
- `tests/test_features_leakage.py` guards this: future data cannot change past features, the current session's path
  never reaches the `pre` features, the `final` stage sees only the session open, global series follow availability,
  and a live row equals the batch row for the same day (train/serve parity). Walk-forward folds are separated by an
  embargo.

Snapshots (features, predictions, data quality, model and code version) are **immutable**: one is written per
(date, stage) in the normal flow, the snapshot and its predictions are committed in **one transaction**, a retry
re-sends the frozen report (restoring the predictions if an old interrupted freeze lost them), and an operator's forced
re-scan adds a new snapshot without ever editing an old one. Evaluation scores only the *effective* (latest) snapshot of
a (date, stage), and a day is scored on its final snapshot, else its preliminary one, so a forced re-scan is never counted
twice. The stored features carry a hash that can be re-verified, the inputs the scan saw (latest source timestamps, arrival
times, counts) are recorded, and `code_version` is `<release>+<fingerprint of the code>`: any frozen report can be traced to
the exact code that produced it (`monster-ai version` prints the running value).

## Models and governance

- **Specialists** (`models/specialists.py`): technical, F&O, global, regime, events, direction. Each is a small
  LightGBM model per target (range thresholds 1/2/3/5 %, "unusual" = range above 1.5x the trailing mean, up/down).
- **Meta model** on out-of-fold specialist outputs plus context; **isotonic calibration** per head; quantile heads
  for the expected range (q10/q50/q90).
- **Walk-forward training** with an embargo, 8 folds, a held-out final block; ranking edge is measured against
  simple baselines (20-day range, ATR, yesterday's range, volume, opening gap, random) with a day-block bootstrap.
- **Gates** (fixed in config, not tuned per run): enough held-out days, ECE <= 0.06, positive Brier skill vs climatology
  and a one-sided 90 % edge over the best baseline. Status is `validated`, `shadow` (calibrated and skilful, edge not
  demonstrated: still useful for research), `rejected` (never activated) or `baseline` (no trained model, heuristic only).
- **Registry**: every version is stored with metrics, gate outcome, feature-schema hash, the code fingerprint and a manifest
  checksum recorded at registration. A challenger replaces the champion only if it matches or beats it on the same gate
  metrics; a corrupt, stale or wrong-feature-version champion is replaced or rolled back automatically, falling back to the
  heuristic baseline if nothing verifies. Daily outcomes never edit a model. A change of `FEATURE_VERSION` (now `fv2`)
  retires the older models: scans use the labelled baseline until the next `train` has produced a model.

```mermaid
flowchart TB
    T0["train job (Sat 02:00 IST, heavy lock)<br/>or: monster-ai train"] --> T1{"free disk above<br/>the critical threshold?"}
    T1 -- no --> TX["fail early, alert, keep the champion"]
    T1 -- yes --> T2["point-in-time training panel<br/>split-adjusted bars, survivorship-aware universe,<br/>special sessions removed"]
    T2 --> T3["walk-forward folds with embargo<br/>8 folds + a held-out final block"]
    T3 --> T4["specialists per target<br/>technical, F&O, global, regime, events, direction"]
    T4 --> T5["meta model on out-of-fold outputs<br/>isotonic calibration, quantile heads"]
    T5 --> T6{"fixed gates<br/>enough days, ECE, Brier skill,<br/>edge over the best simple baseline"}
    T6 -- "all met" --> S1["validated"]
    T6 -- "calibrated and skilful, no edge" --> S2["shadow"]
    T6 -- "not calibrated or no skill" --> S3["rejected - never activated"]
    S1 --> R1["registry: artifact, manifest checksum,<br/>metrics, code fingerprint"]
    S2 --> R1
    R1 --> R2{"first model, champion unhealthy,<br/>higher status, or matches / beats<br/>the champion on the gate metrics?"}
    R2 -- yes --> A1["activate as champion"]
    R2 -- no --> A2["kept as a challenger"]
    A1 --> L1["every scan: verify checksum<br/>and feature version before use"]
    L1 -- "corrupt, stale or wrong feature version" --> L2["roll back to the previous good version,<br/>else the labelled heuristic baseline"]
    EXT["external worker (GitHub Actions)<br/>HMAC-signed bundle in S3"] --> SYNC["model_sync verifies pointer, hash,<br/>signature and every file"]
    SYNC --> R1
```

## Honesty rules

- Probabilities come from calibrated models. The heuristic baseline is always labelled "(heuristic, NOT calibrated)".
- Conviction tiers (STRONG >= 55 %, MODERATE >= 40 %, WATCH >= 25 % calibrated P, each with an uncertainty ceiling) are capped
  at **WATCH unless the model is `validated`**.
- The uncertainty ceilings are **relative, not absolute**. Uncertainty (0 reliable .. 1 unreliable; not a probability) rises with
  the probability itself, because the names most likely to have a large range are the volatile ones with the widest
  intervals and the most specialist disagreement; a fixed ceiling therefore made STRONG unreachable on real data, and one
  pooled ceiling would abstain on every hot day. At training time the model records how uncertain its own held-out daily
  top-15 names were at each calibrated-probability level (`unc_scales.pick_curve`: median and 85th percentile). A live pick is
  judged against names of **similar probability**: STRONG needs uncertainty at or below their median, MODERATE at or below
  their 85th percentile. A pick (or a whole day) that is unusually unreliable for its level therefore still abstains. Models
  without a curve (older artifacts, the heuristic baseline) use `MAX_UNCERTAINTY_STRONG` / `MAX_UNCERTAINTY_MODERATE`. The
  ranking itself is unaffected: it stays by probability.
- When nothing qualifies the report says **"no setup"** and why; it never pads the list. The reason names the actual cause
  (low probability, uncertainty above the model's usual range for its picks, conflicting evidence) with counts.
- A report that would be delivered after the 09:14:30 deadline is withheld (the alert is queued and sent later).
- Ranking uses the calibrated range probability only. The report states that a wide range does not imply a profitable
  trade or a directional close; the direction head is a diagnostic and is near chance in out-of-sample tests.
- Every statistic in the evening report carries its sample size and a 95 % interval (day-block bootstrap, because the picks
  of one day are dependent); below five days it prints "n/a" instead of a number it cannot support. News catalysts show how far
  they can be trusted (rumour, source class, independent sources, conflicts).
- The AI services are advisers, never authorities: the rules always run, an AI answer only adds information (an LLM claim
  needs a quote that appears word for word in the text; Jev must be confident and may not clear a rumour flag or flag an
  official filing; the web check can only discount a story it found denied, never raise anything), a failure falls back to the
  rules, and the report says when a service was degraded. They change which catalysts are shown and how they are labelled,
  **not** the calibrated forecast.

## Scheduling and self-healing

- **Ledger-based scheduler** (`automation/scheduler.py`): one child process at a time (`python -m app.cli run-job`), so
  a leak or crash is contained; sd_notify watchdog; timeouts; preemption of non-critical jobs by critical ones.
- **Idempotent jobs**: locks per job and a shared heavy lock; run keys (`2026-11-02`, `2026-W44`); retries with
  back-off; catch-up for after-close jobs; windows that closed before the first start are skipped silently.
- **Bootstrap** is a resumable 12-step state machine (`bootstrap.state` in the `kv` table). A crash, reboot or
  out-of-memory kill resumes where it stopped; repeated crashes in an optional step skip that step instead of looping.
- **Watchdogs**: per-job RSS ceiling (`JOB_MAX_RSS_MB`), systemd `MemoryHigh/MemoryMax`, service watchdog, a health
  timer that alerts independently of the daemon, a heartbeat that is written to a file as well as to the database (so a
  locked database is distinguishable from a dead daemon), and a self-reported alert when the daemon finds that its previous
  run ended uncleanly.
- **Resource gates**: heavy jobs are deferred while free disk is below the critical threshold or RAM is low, and the light jobs
  behind them still run; training checks the disk before it starts.
- **Health** (`automation/health.py`): data freshness, missing sessions, model age and status, source failures,
  backup age, disk and RAM, storage budget and retention, calendar coverage, delivery failures, performance drift. Alerts are
  de-duplicated and queued until Telegram works again.

```mermaid
stateDiagram-v2
    [*] --> Due: window open, run key not settled, dependencies done
    Due --> Running: the scheduler launches the child process
    Running --> Success: the job returns success
    Running --> Skipped: nothing to do today (closed day, external mode ...)
    Running --> Failed: error, timeout, RSS ceiling or the process died
    Running --> Preempted: a critical job needs the slot
    Preempted --> Due: retried after its interval, no attempt used
    Failed --> Due: back-off elapsed and attempts left
    Failed --> Missed: attempts exhausted, alert job_missed
    Due --> Missed: window closed without a run (jobs without catch-up)
    Success --> [*]
    Skipped --> [*]
    Missed --> [*]
```

```mermaid
stateDiagram-v2
    [*] --> init
    init --> universe
    universe --> calendar_official
    calendar_official --> history
    history --> calendar_inferred
    calendar_inferred --> official_recent
    official_recent --> fno_history
    fno_history --> board_meetings
    board_meetings --> news
    news --> train
    train --> smoke_scan
    smoke_scan --> telegram
    telegram --> complete: init, universe and history are done
    telegram --> incomplete: a required step is not done
    incomplete --> init: the scheduler starts the bootstrap again
    complete --> [*]
```

Each bootstrap step records its own status in `bootstrap.steps`; `init`, `universe` and `history` are required, the rest
can fail or be skipped without blocking the timetable (they are retried by the regular jobs).

```mermaid
flowchart TB
    S["step starts"] --> A{"already done?"}
    A -- yes --> D["skip: already done"]
    A -- no --> B{"crashed twice before<br/>and not required?"}
    B -- yes --> K["mark skipped: earlier attempts<br/>died mid-step (out of memory?)"]
    B -- no --> R["save status running"]
    R --> X["run the step"]
    X -- "done, skipped or failed" --> W["save status and detail"]
    X -- "exception" --> F["status failed, logged with traceback"]
    X -- "process killed: OOM, reboot" --> C["row stays running:<br/>counts as a crash next time"]
```

```mermaid
flowchart TB
    subgraph L1["1. Inside a job"]
        direction TB
        A1["per-job timeout"] ~~~ A2["RSS watchdog:<br/>abort, retry leaner"] ~~~ A3["bounded retries,<br/>back-off, deadlines"]
    end
    subgraph L2["2. Scheduler daemon"]
        direction TB
        B1["ledger: nothing<br/>runs twice"] ~~~ B2["one child at a time,<br/>critical jobs preempt"] ~~~ B3["disk and RAM gates<br/>defer heavy jobs"] ~~~ B4["heartbeat: watchdog,<br/>file, DB row"] ~~~ B5["daemon_restarted<br/>alert"]
    end
    subgraph L3["3. systemd"]
        direction TB
        C1["Restart=always,<br/>WatchdogSec=180"] ~~~ C2["MemoryHigh /<br/>MemoryMax"] ~~~ C3["health timer<br/>every 5 minutes"]
    end
    subgraph L4["4. Operator signals"]
        direction TB
        D1["Telegram alerts,<br/>queued while down"] ~~~ D2["monster-ai health,<br/>logs, diag, storage"]
    end
```

## Network layer: curl_cffi, bulk fetching, rate limits

Every call to a scraped or public data source goes through one client, `ingestion.http.HttpClient` (built by
`HttpClient.from_settings(source, settings)`). Its job is to be a polite, resilient, browser-like client that cannot burn the
09:12 cutoff:

- **Transport: curl_cffi.** libcurl with a real browser TLS / HTTP-2 fingerprint (`HTTP_IMPERSONATE`, default `chrome`), so the
  CDNs and bot managers in front of NSE and Yahoo treat the server like an ordinary browser. If curl_cffi cannot be imported
  or rejects the profile, the client degrades to `requests` (first without the impersonation profile, then without
  curl_cffi) and logs it once. While impersonating, the browser's own `User-Agent` / `Accept-Encoding` headers are never
  overridden (a mismatch with the TLS fingerprint is an easy bot signal). `HTTP_TRANSPORT` can force a transport.
- **Politeness.** A per-source rate limiter spaces requests (`YAHOO_MIN_INTERVAL_SEC`, `NSE_MIN_INTERVAL_SEC`, 0.4 s per RSS
  feed, 6 s for GDELT) with 20 % jitter; a 429 or `Retry-After` doubles that source's spacing (up to a cap) and every success
  eases it back. A circuit breaker fails fast after 6 consecutive failures and half-opens after 5 minutes.
- **Retries.** Statuses 408, 425, 429, 500, 502, 503 and 504 retry with exponential back-off and jitter (or `Retry-After`,
  capped at 60 s), at most `HTTP_MAX_ATTEMPTS`; no attempt starts after the caller's **deadline**, and each attempt's timeout is
  capped to the time left, so the pre-open polls can never push a report past its cutoff.
- **Safety.** HTTPS only; a hard response-size cap (`HTTP_MAX_RESPONSE_MB`, a RAM guard); errors and logs carry host and path
  only, never the query string (it may hold a key); NSE's cookie handshake is re-done automatically on 401/403.
- **Traceability.** One DEBUG line per request, WARNING lines for retries and give-ups, and per-source counters (requests,
  failures, 429s, bytes) that the job runner logs as a `network:` line at the end of every job.

```mermaid
flowchart TB
    CALL["caller: yahoo, nse, news ...<br/>HttpClient.from_settings(source, settings)"] --> HTTPS{"https URL?"}
    HTTPS -- no --> REF["refuse: DataUnavailable"]
    HTTPS -- yes --> BRK{"circuit breaker open?"}
    BRK -- yes --> FAST["fail fast: CircuitOpen"]
    BRK -- no --> RL["per-source rate limiter<br/>spacing + jitter, doubles after a 429,<br/>eases back on success"]
    RL --> DL{"caller deadline reached?"}
    DL -- yes --> DLX["give up, retryable later"]
    DL -- no --> TR
    subgraph TRG["Transport"]
        TR["curl_cffi session<br/>browser TLS and HTTP/2 fingerprint"]
        RQ["requests session<br/>fallback"]
        TR -. "profile rejected: drop impersonation, then curl_cffi" .-> RQ
    end
    TR --> RESP{"status"}
    RQ --> RESP
    RESP -- "2xx or a soft status (304, 404 ...)" --> BODY["stream the body under the size cap"]
    RESP -- "408 425 429 500 502 503 504" --> BACK["Retry-After or exponential back-off with jitter"]
    BACK --> RL
    RESP -- "other 4xx, or attempts used up" --> ERR["DataUnavailable, secret-safe message,<br/>breaker counts the failure"]
    BODY --> STATS["per-source counters -> the job's network line in the log"]
```

**Bulk fetching and batching** keep the request count (and so the rate-limit exposure) low:

| Data | Strategy |
|------|----------|
| Equity end-of-day bars | the **official NSE file** (one request per day, every stock, official OHLC / VWAP / delivery); Yahoo is only the repair path for names the file lacks and for the first history backfill |
| Global series (~33 indices, FX, commodities, rates, VIX) | one Yahoo `spark` probe answers **20 symbols per request**; series whose newest stored bar already equals the vendor's are skipped (two requests instead of ~33); the probe can only skip a fetch, any doubt falls through to the normal fetch, and every series is fetched in full at least every 24 h |
| History backfill | resumable per instrument (`yahoo_history:*` progress), deadline-bounded chunks, one full download per symbol |
| F&O | one UDiFF zip per day (all contracts), parsed once |
| RSS feeds | **conditional GET** (`ETag` / `Last-Modified`): an unchanged feed costs one tiny 304; one unconditional fetch every 6 h; a feed that fails 6 times is parked for 24 h |
| NSE API | one pre-open call returns the whole book; GIFT Nifty comes from the market-status call already needed for the session state |
| GDELT | minimum spacing 6 s, a few queries per run |

```mermaid
flowchart TB
    U["universe ~250 stocks"] --> OFF["NSE archive: ONE file per day<br/>official bars for every stock"]
    OFF --> DB[("SQLite")]
    OFF -. "names missing from the file" .-> REPAIR["Yahoo chart request,<br/>only for those names"]
    REPAIR --> DB
    GS["~33 global series"] --> SP["Yahoo spark probe<br/>20 symbols per request"]
    SP --> CH{"newest bar already stored<br/>and fetched in full within 24 h?"}
    CH -- yes --> SK["skip: no per-series request"]
    CH -- no --> CHART["chart request for that series"]
    CHART --> DB
    FEEDS["RSS feeds"] --> CG["conditional GET"]
    CG -- "304 unchanged" --> SK2["nothing downloaded"]
    CG -- "200 changed" --> NI["parse, dedup, store"]
    NI --> DB
```

## Exchange calendar for any month and year

The calendar (`automation/calendar.py`) never needs a code change when a year turns over, and works with or without
network access:

1. weekends are always closed;
2. the shipped seed (`data/seeds/nse_holidays.json`) and NSE's live holiday API (refreshed daily at 05:30 IST) are the **official**
   lists; a year counts as officially covered once it has at least 8 entries (NSE lists 14-20 trading holidays a year);
3. for a year without an official list (NSE publishes the next year's in December) a **rule layer** applies the holidays that
   can be derived with certainty - the fixed-date national holidays (26 Jan, 14 Apr, 1 May, 15 Aug, 2 Oct, 25 Dec) when they fall
   on a weekday, and Good Friday from the Gregorian Easter algorithm - flagged `source='rule'`. Lunar-calendar holidays
   (Holi, Diwali, Eid ...) cannot be derived and are deliberately **not guessed**;
4. the **live confirmation**: before a final report goes out in such a year, NSE itself is asked. If it answers but
   publishes no pre-open book after 09:06 IST, the day is recorded as an inferred closure, no report is sent and the
   operator gets a `holiday_inferred` alert. An unreachable NSE proves nothing, so the scan simply goes on;
5. days on which the Nifty index has no bar are inferred as closed retroactively, and an **operator override file**
   (`data/seeds/special_sessions.json`, or `calendar_overrides.json` in the data directory, which survives upgrades) can
   re-open a day (Muhurat and other special sessions) or close one at short notice.

Rule rows disappear the moment the official list arrives; official rows are never overwritten by rules, and a holiday NSE
later removes from a future date is dropped. Health warns 21 days before the official list ends (`calendar_horizon`), after it
lapsed (`calendar_unofficial`) and when the refresh keeps failing (`calendar_stale`, `calendar_never_refreshed`); the report
itself says "holidays are rule-based" while it relies on the rule layer.

```mermaid
flowchart TB
    Q["is_trading_day(date)"] --> W{"Saturday or Sunday?"}
    W -- yes --> CL["closed"]
    W -- no --> ROW{"row in calendar_days?"}
    ROW -- yes --> USE["use the row: holiday, or special session<br/>with its own open and close time"]
    ROW -- no --> OPEN["open: regular session 09:15 to 15:30 IST"]
    subgraph WRITERS["Who writes calendar_days"]
        S1["seed: data/seeds/nse_holidays.json<br/>(official, shipped)"]
        S2["nse_api: holiday API, refreshed daily 05:30<br/>(official, replaces seed, rule and inferred rows)"]
        S3["rule: fixed-date holidays and Good Friday<br/>for years without an official list"]
        S4["inferred: no index bar, or no pre-open book<br/>on the day (live confirmation)"]
        S5["override: special_sessions.json and<br/>calendar_overrides.json (never replaced)"]
    end
    WRITERS --> ROW
    H["health: horizon, unofficial, stale, never refreshed"] -.-> WRITERS
```

## Time zones

A server in any time zone produces the same results. The rules, enforced by tests:

- **Storage is UTC.** Every stored instant is an ISO-8601 UTC string (`...Z`); the clock hands out timezone-aware instants only.
- **Market logic is IST, explicitly.** Session dates, schedule windows, scan cutoffs and deadlines are computed with
  `ist_date`, `at_ist` and friends from `app/clock.py` using `Asia/Kolkata` (a fixed +05:30 offset is the fallback if the
  tz database is missing). India has no daylight saving, so nothing shifts twice a year.
- **The process zone is pinned to UTC** at start-up (`pin_process_timezone` in `cli.main()`), so even a stray `localtime()`
  cannot differ between servers; no systemd timer depends on the system zone (the scheduler is ledger-based, the health timer
  is monotonic).
- **Display only:** `MONSTER_TIMEZONE` (default `Asia/Kolkata`) sets the zone of log stamps (`2026-11-02T08:30:05.000+0530`)
  and nothing else.
- **Guards:** an AST test fails the build on any API that silently reads the server's zone (naive `datetime.now()`, `utcnow()`,
  `date.today()`, `time.localtime()`, `.astimezone()` without an argument, SQLite `'now'` / `'localtime'`; two documented
  exceptions), and a matrix test computes exchange dates, session windows, the job timetable, holidays, retention cut-offs and log
  stamps in fresh interpreters whose process zone is UTC, US Pacific (DST), India, New Zealand (DST) and UTC+14, requiring
  identical results equal to values worked out by hand.

```mermaid
flowchart TB
    OS["server clock and zone<br/>any TZ, DST, NTP steps"] --> PIN["pin_process_timezone()<br/>process zone forced to UTC"]
    PIN --> CLK["app.clock<br/>aware instants + monotonic time"]
    CLK -->|"stored"| DB[("SQLite<br/>UTC ISO-8601 with Z")]
    CLK -->|"market logic"| IST["ist_date, at_ist<br/>Asia/Kolkata (+05:30 fallback)"]
    IST --> USE["calendar, schedule windows,<br/>scan cutoffs, deadlines"]
    CLK -->|"display only"| LOGZ["log stamps in MONSTER_TIMEZONE"]
    GUARD["tests: AST guard against naive datetimes<br/>and a five-zone subprocess matrix"] -.-> CLK
```

## Storage and bounded retention

Nothing may grow forever. The nightly `housekeeping` job (23:50 IST) applies `automation/retention.py`, in small
foreign-key-safe transactions that never starve the scheduler's heartbeat of the single SQLite writer:

| What | Rule (default) | Setting |
|------|----------------|---------|
| raw news and filing text | deleted after 45 days, structured events kept | `NEWS_RETENTION_DAYS` |
| structured events | 1100 days after their last update | `EVENT_RETENTION_DAYS` |
| pre-open and live quotes | 400 days; every pre-open poll is kept 14 days, then only the last one per symbol and day | `QUOTE_RETENTION_DAYS` |
| daily bars, F&O rows | `HISTORY_YEARS` + 1 years (the oldest data ever read) | `HISTORY_YEARS` |
| snapshots, predictions, outcomes, metrics, graph edge history | 1825 days (immutable: pruned explicitly, children before parents) | `RECORD_RETENTION_DAYS` |
| job runs, deliveries, alerts | 180 days (delivered alerts: half) | `OPS_RETENTION_DAYS` |
| backups | 7 daily + 4 weekly **and** a total size budget | `BACKUP_KEEP_*`, `BACKUP_MAX_MB` |
| logs | 30 days within a 150 MB budget; today's files are never removed | `LOG_RETENTION_DAYS`, `LOG_MAX_MB` |
| cache, temp, quarantined restores, diagnostic bundles | 200 MB / 14 days, 1 day, 14 days, newest 5 | constants |
| the whole data directory | 4000 MB budget; above it the optional data is trimmed harder | `DATA_MAX_MB` |

Under **disk pressure** the optional windows shrink to half (free disk below the warning threshold) or a quarter (below the
critical threshold), never below a week; the immutable records and the price history are shortened only by their own window.
`VACUUM` runs only when it is due (Sundays, pressure, or forced), worthwhile (at least 32 MB or 15 % free pages) and the disk has
about three times the database size free. The result of every run is stored (`retention.last`), `monster-ai storage` shows the
usage and `--prune [--dry-run] [--vacuum]` applies it on demand, and health warns at 80 % of the budget, when the WAL file
exceeds 256 MB, or when the retention has not run for three days.

```mermaid
flowchart TB
    HK["housekeeping job, 23:50 IST daily"] --> RET["run_retention"]
    RET --> PR["disk pressure from free space<br/>ok x1, warn x0.5, critical x0.25<br/>(over the data budget: x0.5)"]
    PR --> TB["table policies, children first<br/>outcomes, predictions, snapshots, quotes, news events,<br/>alerts, deliveries, job runs, metrics, F&O, bars,<br/>graph edge history"]
    PR --> KV["kv rules: evaluation summaries,<br/>fetch markers, a stale prune flag"]
    PR --> FS["file sweeps: cache, tmp, backup leftovers,<br/>quarantine, diagnostic bundles, logs"]
    PR --> BK["backups: 7 daily + 4 weekly<br/>within the total size budget"]
    TB --> DBM["maintain_database<br/>WAL checkpoint, PRAGMA optimize,<br/>VACUUM when due, worthwhile and safe"]
    KV --> DBM
    DBM --> REP["retention.last in kv<br/>monster-ai storage report"]
    FS --> REP
    BK --> REP
    REP --> HLT["health: storage budget, WAL size,<br/>retention stale or erroring -> Telegram alert"]
```

## Logging and diagnostics

Every process (daemon, each job child, CLI commands, the dashboard) logs through `app/logging_setup.py`:

- **Daily files, never renamed.** `logs/<role>-YYYY-MM-DD.log` (roles: `daemon`, `jobs`, `cli`, `health`, `train`, `bootstrap`,
  `dashboard`), at most 25 MB x 4 files per day with a drop-count marker, mode 0640. A shared `errors-YYYY-MM-DD.log` collects
  WARNING and above of **all** processes with their role and pid, so one file answers "what went wrong today". Concurrent
  processes can append safely (no rotation renames), and logrotate is deliberately restricted to redirected `.out` files and the install log.
- **Traceable.** Stamps carry milliseconds and the offset; each job line carries its tag `[<job> <run key>#<attempt> r<run id>]`; the
  job runner logs start, duration, peak memory and a per-source `network:` line; the daemon logs every finished job with its
  duration; unexpected exceptions are logged with tracebacks (token-bearing paths log only the place, `exception_site`).
- **Crashes are captured:** `sys.excepthook`, thread and unraisable hooks, and `faulthandler` (native crashes) write to the log
  and to `fault-<role>.log`; warnings from libraries are captured too.
- **Secrets are redacted** in the formatter, so no handler can leak them; `LOG_FORMAT=json` switches to one JSON object per line.
- **Operator tools:** `monster-ai logs [--role R] [--date D] [-n N] [--level L] [--job J] [--grep T] [--since 90m] [--follow]
  [--list]` reads, filters (with multi-line tracebacks kept together) and follows the files in bounded memory;
  `monster-ai diag` writes a redacted `diag-<UTC>.tar.gz` (versions, offline doctor, settings, health, storage, database summary,
  log tails) to attach to a bug report.

```mermaid
flowchart TB
    subgraph PROCS["Every process"]
        D["daemon"]
        J["job children"]
        C["CLI commands"]
        W["dashboard"]
    end
    PROCS --> LG["logging_setup: redaction, job context,<br/>display-zone stamps"]
    LG --> F1["logs/ROLE-YYYY-MM-DD.log<br/>25 MB x 4 files per day"]
    LG --> F2["logs/errors-YYYY-MM-DD.log<br/>WARNING and above, all roles"]
    LG --> JD["stderr to journald"]
    CR["crash hooks: excepthook, threads,<br/>unraisable, faulthandler"] --> F1
    CR --> FL["logs/fault-ROLE.log"]
    F1 --> LV["monster-ai logs:<br/>level, job, text, time, follow"]
    F2 --> LV
    F1 --> DG["monster-ai diag:<br/>redacted bundle"]
    F2 --> DG
    HKJ["housekeeping: prune_logs<br/>30 days and 150 MB"] -.-> F1
```

## Degradation matrix

| Failure | Behaviour |
|---------|-----------|
| NSE API blocked / pre-open empty | final scan uses the `pre` model for names without a pre-open price and says so; if the book never arrives the report is still sent, marked |
| Yahoo down or rate limited | official bhavcopy first; global block is withheld only when >= 6 critical series are stale, otherwise degraded with warnings; the client slows that source down and a circuit breaker stops hammering it |
| curl_cffi unusable (missing wheel, rejected profile) | automatic fallback to `requests`, logged once; some providers may then block the client |
| Previous session incomplete | repair from the official file; if coverage stays low the data-quality gate **blocks** the scan, alerts and (on the last attempt of the final scan) tells you why |
| Vendor bars still provisional | the official file replaces them in the morning job and again, for the last three sessions, in the nightly housekeeping |
| Split / bonus with an ex-date today | the name is scored without its meaningless opening gap; the vendor is asked until the split is recorded, the operator is told after six sessions |
| Holiday list of a new year not published | rule-based holidays plus the live pre-open confirmation; `holiday_inferred` alert and no report on a closed day |
| RSS feeds failing | news catalysts reduce to "coverage reduced" in the report; scoring continues; a failing feed is parked for 24 h |
| Groq LLM or Jev unavailable, unconfigured, rate-limited or rejected | the rules alone; the circuit opens for the run (at once on a rejected key), a quota starts a cooldown that survives between runs, the report's coverage note names the service, `source_health` warns after repeated failures |
| Web check unavailable (plan or model refuses the tool, slow or failing search) | the item is extracted and judged without it; the web check has its own circuit and a 6 h back-off, so the extraction is untouched; `source_health` (`llm_web`) warns after repeated failures |
| No active model, corrupt artifact, wrong feature version | automatic rollback, then heuristic baseline with the honest label |
| A model file cannot be read (I/O error) | the champion is kept and the labelled baseline serves that scan only (`model_load_failed`); only definite damage (missing or altered files, wrong feature version) retires a model |
| gpg missing while backup encryption is configured | the local backup stays, unencrypted, and is **not** uploaded off-box; `backup_encryption_failed` alert and a health warning until it is fixed |
| Telegram down | reports and alerts are queued in the ledger and flushed when it returns; late final reports are withheld |
| Process killed / reboot | systemd restarts the daemon; the ledger resumes; SQLite WAL recovery; `daemon_restarted` alert |
| Database locked while the daemon lives | the heartbeat file proves liveness; `heartbeat_db_stale` warns instead of a false "daemon dead" |
| Disk low | emergency cleanup of caches, logs and temp files; heavy jobs deferred and training refused; critical alert; retention windows shrink |
| Storage creeping up | nightly retention keeps every table, folder and cache bounded; health warns at 80 % of the data budget |
| Time glitch (NTP step) or a server in another zone | windows are evaluated from the ledger and the IST clock; the process zone is pinned to UTC; nothing runs twice |

## External training and off-box storage

Optional. A GitHub Actions worker ([.github/workflows/train.yml](../.github/workflows/train.yml)) restores the
server's newest off-box backup as its training data (or collects history itself), trains with a larger budget, and
publishes **HMAC-signed bundles** (`models/<stage>/<version>.zip` + a signed `latest.json` pointer). The server
(`TRAINING_MODE=external`) only pulls: it verifies the pointer, the archive hash and the manifest signature
*before* extracting anything, re-verifies every file, rejects a bundle that is not `shadow`/`validated`, built for another
feature version, or whose metadata disagrees with its manifest, and then registers it as a normal challenger.
Off-box backups use the same S3-compatible store and are pruned to the local retention.

## Deployment topology

```mermaid
flowchart TB
    PC["your PC<br/>deploy_remote.ps1 or .sh"] -->|"ssh + tar, env file shredded afterwards"| INST
    subgraph SRV["Ubuntu 22.04/24.04 or Debian 12 server"]
        INST["install.sh<br/>idempotent, 14 steps, atomic swap, rollback"]
        subgraph SD["systemd units"]
            direction TB
            U1["monster-ai.service<br/>daemon, Type=notify, WatchdogSec=180"] ~~~ U2["monster-ai-bootstrap.service<br/>first-run setup, resumable"] ~~~ U3["monster-ai-health.timer<br/>every 5 minutes"] ~~~ U4["monster-ai-alert@.service<br/>OnFailure alert"] ~~~ U5["monster-ai-dashboard.service<br/>optional, loopback"]
        end
        subgraph FS["Filesystem"]
            direction TB
            OPT["/opt/monster-ai<br/>(+ .prev for rollback)"] ~~~ ETC["/etc/monster-ai/monster.env<br/>root:monster 0640"] ~~~ VAR["/var/lib/monster-ai<br/>database, backups, models, cache"] ~~~ LOGD["/var/log/monster-ai<br/>daily logs"]
        end
        INST --> SD
        SD --> FS
        U1 -->|"spawns"| JOBS["job processes<br/>python -m app.cli run-job NAME"]
    end
    JOBS -->|"HTTPS only"| NET["Yahoo, NSE, RSS, GDELT, Telegram,<br/>optional S3, Groq and Jev"]
```

## Security

- No orders, no broker keys. Secrets only in `/etc/monster-ai/monster.env` (`root:monster 0640`); logs and diagnostic bundles redact them.
- Dedicated unprivileged `monster` user; every unit runs with `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`,
  `PrivateTmp`, an empty capability set and IP/Unix sockets only; the dashboard binds to loopback (a token is
  required otherwise).
- Model artifacts are LightGBM text + JSON; bundles are signed and extracted with zip-slip / zip-bomb protection.
- All inbound data is validated (schema, ranges, dates, size limits); SQL is parameterised; HTTP clients are HTTPS-only with
  timeouts, size limits, bounded retries and secret-free error messages.
- Outbound connections: the data providers, Telegram, and - only if configured - the object store, Groq (or another LLM endpoint) and
  Jev. Only public news text leaves the server for the AI services; their keys are secrets like the others (`GROQ_API_KEY` is only
  ever sent to a URL that is exactly `https://api.groq.com...`: look-alike hosts and URLs with credentials or backslashes are
  refused or treated as another provider, and a plain-http endpoint is accepted for loopback hosts only). The web check's searches are made by Groq from its own servers: the server only sends the
  request to Groq. Text that comes back from the web is untrusted: it is reduced to a bounded, validated brief before it is used.

## Resource budget

Target: 1 GB RAM, 1 vCPU, 8 GB disk. The daemon is a light process (it never imports pandas, numpy or requests);
jobs run as short-lived children inside the service's cgroup (`MemoryHigh=750M`, `MemoryMax=880M`, job RSS ceiling
720 MB). A morning scan took about 11 s on a development PC. Training is the only heavy job: it runs weekly in a
lighter profile on small servers (60 000 rows per booster fit, five years of history for a 221-stock universe) or
externally. Measured on the real data with the light profile: the feature build peaks at about 640 MB resident and the
model training at about 565 MB. Should a run still die from memory pressure, the retry trains a leaner profile (one year
less history and half the fit sample per failed attempt). The installer adds a small swap file as a safety net; swap is not
a substitute for RAM. Disk is bounded by the retention rules above (4 GB data budget by default, well inside 8 GB).

## Design decisions

| Decision | Why |
|----------|-----|
| Modular monolith, one SQLite file | one process owner of state, trivial backup/restore, enough for ~250 symbols |
| Native systemd, **no containers** | 1 GB budget; fewer moving parts; journald + systemd already give restart, limits and logs |
| Ledger instead of cron | a restart or clock step cannot skip or duplicate a job; missed work is visible |
| One feature path for training and live | removes train/serve skew, the usual source of silent leakage |
| Immutable snapshots, one effective snapshot per (date, stage) | the evaluation can always be traced to what was known and sent, and a forced re-scan cannot be counted twice |
| Calibrated probabilities + honesty gates | a ranking is only useful if "40 %" means 40 %; unvalidated models are visibly capped |
| Shadow-first | the edge must be demonstrated out of sample before anyone risks money |
| curl_cffi behind one `HttpClient` | scraped endpoints sit behind bot managers; one place owns impersonation, spacing, retries and deadlines |
| Groq and Jev optional, advisory, behind one set of rails (`news_intelligence/remote.py`) | the rules stay deterministic and auditable; an AI answer can only add validated information, and a provider outage, a rate limit or a bad key costs coverage, never a delayed or corrupted report |
| UTC storage, IST logic, pinned process zone | identical behaviour on any server, in any zone, in any season |
| Explicit retention for everything | a 8 GB disk must last years without a human; every table, folder and cache has a bound and a test |
| Self-reporting operations (heartbeat file, restart alert, health timer) | systemd's `OnFailure=` practically never fires for an always-restarted service, so the system reports its own failures |

## Deviations from the blueprint

The full requirement-by-requirement mapping, with the status of every item, is in
[BLUEPRINT_TRACEABILITY.md](BLUEPRINT_TRACEABILITY.md). The deliberate deviations:

| Blueprint says | What was built | Why and consequence |
|----------------|----------------|---------------------|
| "Start with LightGBM, XGBoost, CatBoost and statistical baselines" | LightGBM specialists and meta model, plus the heuristic and simple statistical baselines | one boosting library keeps the wheel set small and the artifacts text-only; the baselines are what the model must beat |
| "FinBERT or other financial NLP models; sentence embeddings for event clustering" | rule-based event extraction with optional hosted AI (Groq LLM, Jev), lexical polarity, SimHash / TF-IDF clustering | no model download or GPU on a 1 GB server; deterministic and auditable. News is evidence in the report, not a model input |
| "systemd + APScheduler or cron" | systemd plus an in-house ledger scheduler (`automation/scheduler.py`) | restart-safe, catch-up, nothing runs twice, no extra dependency |
| deployment folder with "service files, Docker, backups" | native systemd, **no containers of any kind** | the operator requires it, and the RAM budget agrees |
| VWAP features | a daily proxy from the official file's average price | no tick or long intraday history is available from free sources |
| pre-open "indicative price and imbalance where the feed provides it" | indicative and final price are used; order quantities (imbalance) are parsed and stored but not shown or used | the free NSE API has no history to train on and a new feature would change the schema |
| licensed market-data feeds | free public endpoints (Yahoo, NSE, RSS, GDELT) | no paid keys; they can change or block, so every source has a documented failure behaviour and a backup source |
| trading / execution layer | none: shadow mode only | the system measures range prediction, not profitability |
