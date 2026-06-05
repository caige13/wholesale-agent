# Design choices

Notes on the decisions behind the wholesale order agent: what I chose, why, and the
tradeoffs I accepted.

## Pricing and stock

Prices and live stock stay out of the RAG index. They change too often, and any small
price change would force a re-embed. Instead the agent reads them from the supplier at
request time through the SupplierGateway port (get_price and check_inventory). The
catalog only holds stable product facts. CatalogItem deliberately carries no price or
stock.

For the demo I assume the supplier exposes this over an API. If a real supplier had no
API, the fallback would be to build an order draft and hand the user off to the
supplier's own checkout.

Tradeoff: a live call on every inventory check adds latency and a dependency on the
supplier being reachable. I accept that because serving a stale price or an item that's out of stock on an order is
worse than a short wait.

## UI: Gradio

Gradio gives a clean chat interface out of the box, which is enough for a demo and
keeps all the logic on the agent side.

## Planning first

I worked out a spec and plan with AI before writing code. It leans more toward
spec-driven development than strict TDD, though the deterministic core was still built
test first.

## Architecture: ports and adapters

I chose a clean-architecture style. Ports and adapters let me swap implementations for
the same role without touching the business layer, for example Gemini or OpenAI behind
the chat model, or a different embedder behind the catalog. The app package is the
business layer, the domain package holds pure logic, and adapters hold the
infrastructure.

## Single intent for v1

I started with a single-turn graph that classifies one intent. Forcing one intent per
message is limiting when a user asks a question and places an order in the same
message. Multi-intent adds a complexity tax, so v1 stays single-intent and I would
revisit it later.

What I built for the question case: the intent router sends a question to a dedicated
QA subgraph. The order branch runs its own retrieval, but only to resolve SKUs, not to
answer questions. A mixed message is routed to one branch, so it is either answered or
ordered, not both.

Tradeoff: single-intent keeps the graph simple and predictable. The cost is that
"answer this and also add that" in one turn only does one of the two.

## The order path is deterministic

An earlier version of this note described a guard that would stop the order nodes from
reaching "cart op tools" when there were no cart ops. That is not how it ended up. The
order path has no tools at all. Every step (resolve, companions, inventory, validate,
apply) is plain deterministic code. When parse produces zero cart ops, each step
iterates an empty list and changes nothing.

The clarify-versus-draft decision is a separate gate after apply. It looks at the final
line items and asks for clarification when something is low confidence or carries a
blocking flag, otherwise it drafts.

Tradeoff: keeping the order path tool-free makes it predictable and cheap to test with
a scripted model. The downside is that the no-op steps still appear as nodes in the
trace even when there is nothing to do. See "Open ideas" for whether to prune them.

## Question answering

A question turn passes the whole message to the QA node, which is specialized for
answering. If the message also contains an order, the QA node still just answers. This
behavior is covered by the eval set.

## Eval judges

The eval uses a different model for the judge (OpenAI) so it never grades its own work.
The deterministic parts of scoring (extraction, clarification, submission, escalation)
are pure functions. They are unit-tested with a FakeJudge and no real model. A test
feeds a known cart and an expected row and asserts the score, including the
partial-credit range between 0 and 1. Only answer faithfulness needs a real judge.

## Eval runner

The flow is: load the dataset, run the real agent on each row, apply the judges to each
result, then aggregate. The local runner prints the per-metric averages. The
LangSmith-native runner syncs the same dataset and runs the same judges in the platform,
so experiments are versioned and comparable without the two paths drifting.

## Companions instead of "needs lids"

Originally we had "Needs lids" which was a single example that doesn't scale out. Therefore, I replaced the hard-coded 
"needs lids" idea with a generic "needs companion" flag. It supports upsells and pairings across suppliers, not only lids.
The example: the LLM adds a deli container, the supplier says lids are a separate order, so the agent offers to add
the matching lids as a friendly reminder. A lid is a companion of the container.

The design supports multiple companions per item. The LLM is fed the history and a
pre-selected list of companion offers for that item, and it picks which ones the user
wants. So the model acts as a chooser over a closed set, not a free-text parser.
Deterministic code maps the chosen names back to SKUs and sizes the quantities.

## Aliases as data

Aliases live in the catalog and are embedded alongside the product, so "salsa cup to 2oz
portion cup" is a data row, not a code branch. Adding products or suppliers is a data
change. I do not enumerate every phrasing. The embedding retriever generalizes, exact
aliases are curated shortcuts for common or ambiguous cases, and when the agent is
genuinely unsure it asks.

**future work**: the vision here is it can be specialized for the user, and the AI can learn to link these aliases to
a product for the supplier. This would require user association and be a pretty big lift, but that's the vision.

## Supplier API (mocked)

The check_inventory node calls supplier.get_price(sku) and supplier.check_inventory(sku)
rather than keeping prices in RAG. It raises an OUT_OF_STOCK flag when the supplier has
none. The draft node calls supplier.submit_order(items). A place_order flag flows through
the state and decides whether a turn drafts or submits.

## Typos

The LLM corrects obvious typos and rewrites the phrase into the wording the embedder
expects. This keeps the match deterministic while still giving the user some forgiveness
on spelling.

## How the LLM is used

The LLM has two jobs here, and neither is to run the business logic.

First, normalization. It turns messy human input into keyed, structured output that
deterministic code can act on. Typos become the catalog's standard wording, a spoken
phrase becomes a product the embedder can match, and a free-text reply becomes a set of
choices from a known list. The model never invents a SKU or a quantity. It hands clean,
structured inputs to the deterministic path.

Second, judgement over a closed set. When a decision needs language understanding, like
which offered companions the user agreed to, the model picks from options I give it
rather than producing free text. Code then maps the picks back to SKUs and does the math.

Everything that touches the cart, pricing, validation, and submission stays
deterministic. The LLM makes the human input legible, and the code makes the decisions
that have to be correct. This keeps behavior testable with a scripted model and keeps the
costly, non-deterministic part at the edges.

## Eval dataset detail

When a row should not ask for clarification, I avoid using an item that has a companion,
because a companion item would trigger the offer and therefore a question.

## Rate limiter

There is a client-side rate limiter because the demo uses the free Gemini tier, which
allows only a handful of requests per minute for our model. The exact rate is
configurable through settings. It matters most during an eval run that fires many calls
in a row.

## Streaming

I stream the turn to the UI. The agent exposes stream_run, and the UI shows which node
is running and then the answer as it is written.

An earlier version of this note argued for plain invoke and against streaming, on the
grounds that it only saves a few seconds and adds complexity. I reversed that. The
responsiveness is better for UX. The user sees progress instead of waiting for the whole turn,
and the order path reply is still composed deterministically, so only the question
answer actually streams token by token.

Tradeoff: streaming adds some plumbing for event translation and frame handling. In
return the chat feels live.

## Open ideas

Bypass empty order steps. Today the no-op steps run as nodes even when parse produced
nothing to do, so they show up in the trace. A conditional edge could route around them
to keep the trace clean. The catch is that "zero cart ops" is not the same as "nothing
to do". Accepting a companion offer and placing the order both arrive with zero new cart
ops and still need to run. So a safe bypass would only fire when there are no cart ops,
no accepted companions, and no place-order signal, and it would route to draft, not to
clarification. The benefit is mostly trace readability rather than speed, since the steps
already do no real work.

Item memory. The resolver can prefer a learned item-to-SKU mapping, and that path is
unit-tested, but nothing populates it yet, so "the usual" does not resolve. Wiring a
writer that learns per-restaurant mappings would make reorder work. Until then the agent
correctly asks for clarification on a reorder.

Multi-intent. Handle a message that both asks a question and places an order in the same
turn, instead of routing it to one branch.

PII in traces. The redact step produces a scrubbed clean_message and records the PII
types found, but a turn's run inputs still carry the original raw message. For a
production service no PII should reach LangSmith. The trace boundary would mask any PII
to a type descriptor like [SSN] or [Phone number] on the way out, so a reviewer sees the
shape of what was redacted and never the value.