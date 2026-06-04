"""Static UI assets for the Gradio shell — font link, masthead, grain, CSS, a11y JS.

Plain string constants with no gradio dependency, kept out of :mod:`.app` so the
shell module stays focused on layout/wiring. The aesthetic is a warehouse packing
slip: warm paper, charcoal ink, one terracotta ink-stamp accent.
"""

from __future__ import annotations

# Fraunces is loaded via a <link> here rather than an @import in the stylesheet:
# Gradio injects css= through a constructed stylesheet, which strips @import rules.
_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&display=swap\">"
)

_MASTHEAD = (
    _FONT_LINK
    + '<div class="masthead">'
    '<div class="masthead__mark">▦</div>'
    '<div class="masthead__txt">'
    '<div class="masthead__title">The Order Desk</div>'
    '<div class="masthead__sub">restaurant supply · conversational ordering</div>'
    "</div>"
    '<div class="masthead__stamp">Draft</div>'
    "</div>"
)

# Faint fractal-noise paper grain, inlined as an SVG data URI (kept as one constant
# so the stylesheet line below stays readable).
_GRAIN = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence "
    "type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E"
    "%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.035'/%3E"
    "%3C/svg%3E"
)

_CSS = """
:root {
  --paper:#f4efe2; --paper-2:#efe7d6; --card:#fbf7ee;
  --ink:#211c15; --ink-soft:#6f6149; --line:#d9ccb1;
  --stamp:#c0432b; --stamp-deep:#9a3320; --olive:#5b6a39;
}

/* This is a light paper design — pin Gradio's theme tokens to it so the app
   renders the same whether the browser/theme is in light or dark mode (otherwise
   dark mode paints the chat and panels near-black). */
.gradio-container, .gradio-container.dark, .dark {
  --body-background-fill: var(--paper) !important;
  --background-fill-primary: var(--card) !important;
  --background-fill-secondary: var(--paper-2) !important;
  --block-background-fill: var(--card) !important;
  --block-border-color: var(--line) !important;
  --panel-background-fill: var(--card) !important;
  --input-background-fill: var(--card) !important;
  --border-color-primary: var(--line) !important;
  /* The chat bubble's fill/border come from the accent tokens; pin them to the
     paper palette too or Gradio's default --color-accent-soft (a pale near-white)
     bleeds through on the user bubble as a "white box". */
  --color-accent-soft: var(--paper-2) !important;
  --border-color-accent: var(--line) !important;
  --border-color-accent-subdued: var(--line) !important;
  --body-text-color: var(--ink) !important;
  --body-text-color-subdued: var(--ink-soft) !important;
  --block-label-text-color: var(--ink-soft) !important;
  --block-title-text-color: var(--ink) !important;
  color-scheme: light;
}

.gradio-container, .gradio-container .prose {
  background: var(--paper);
  color: var(--ink);
}
.gradio-container {
  background-image:
    radial-gradient(circle at 18% 0%, rgba(192,67,43,.05), transparent 42%),
    url("__GRAIN__");
  max-width: 1180px !important; margin: 0 auto;
}
footer { display:none !important; }

/* Masthead */
.masthead {
  display:flex; align-items:center; gap:18px;
  padding:22px 26px; margin-bottom:8px;
  background:linear-gradient(180deg,var(--card),var(--paper-2));
  border:1.5px solid var(--ink); border-radius:4px;
  box-shadow:6px 6px 0 rgba(33,28,21,.12);
  position:relative;
}
.masthead__mark {
  font-size:30px; line-height:1; color:var(--stamp);
  border:1.5px solid var(--ink); padding:8px 10px; border-radius:3px;
  background:var(--paper);
}
.masthead__title {
  font-family:'Fraunces',Georgia,serif; font-weight:600;
  font-size:30px; letter-spacing:-.01em; line-height:1;
}
.masthead__sub {
  font-family:'JetBrains Mono',monospace; font-size:11px;
  text-transform:uppercase; letter-spacing:.22em; color:var(--ink-soft);
  margin-top:7px;
}
.masthead__stamp {
  margin-left:auto; align-self:flex-start;
  font-family:'JetBrains Mono',monospace; font-weight:700; font-size:12px;
  text-transform:uppercase; letter-spacing:.18em; color:var(--stamp);
  border:2px solid var(--stamp); border-radius:4px; padding:5px 12px;
  transform:rotate(4deg); opacity:.85;
}

/* Chat window — white card, black outline, ink text (regardless of theme mode) */
#deskchat, #deskchat * { background-color:transparent; color:var(--ink) !important; }
#deskchat {
  border:1.5px solid var(--ink) !important; border-radius:4px !important;
  background:var(--card) !important;
  box-shadow:5px 5px 0 rgba(33,28,21,.10);
}
/* In Gradio 6 the role class (.user/.bot) IS the bubble; .message is a wrapper and
   .message-content the inner text box. So the bubble border/fill must target the
   role element (and its inner content) — a `.user .message` descendant doesn't
   exist, which is why the bubbles previously fell back to the near-white card and
   read as a "white box". These need !important to beat the catch-all transparency
   above. */
#deskchat .user, #deskchat .bot,
#deskchat .message.user, #deskchat .message.bot {
  border-radius:4px !important; border:1px solid var(--line) !important;
  font-size:15px;
}
#deskchat .user, #deskchat .message.user, #deskchat .user .message-content {
  background:var(--paper-2) !important;
}
#deskchat .bot, #deskchat .message.bot, #deskchat .bot .message-content {
  background:var(--paper) !important;
}

/* Input + buttons */
/* Gradio wraps the textbox in a .form div with its own tan background + border;
   flatten it so the input is a single clean cream box with one black outline. */
.gradio-container .form:has(#askbox) {
  background:transparent !important; border:none !important; box-shadow:none !important;
}
#askbox, #askbox * { background:var(--card) !important; }
#askbox {
  border:1.5px solid var(--ink) !important; border-radius:4px !important;
  box-shadow:none !important; padding:0 !important;
}
#askbox textarea {
  font-size:15px !important; border:none !important; box-shadow:none !important;
  color:var(--ink) !important; padding:11px 13px !important;
}
#sendbtn {
  background:var(--stamp) !important; color:#fbf7ee !important;
  border:1.5px solid var(--stamp-deep) !important; border-radius:4px !important;
  font-family:'JetBrains Mono',monospace !important; font-weight:700 !important;
  text-transform:uppercase; letter-spacing:.12em; font-size:13px !important;
  box-shadow:3px 3px 0 rgba(33,28,21,.18);
}
#sendbtn:hover { background:var(--stamp-deep) !important; }
#newbtn {
  background:transparent !important; color:var(--ink-soft) !important;
  border:1.5px dashed var(--line) !important; border-radius:4px !important;
  font-family:'JetBrains Mono',monospace !important; font-size:12px !important;
  text-transform:uppercase; letter-spacing:.1em;
}

/* The packing slip — shared card chrome for the static (.slip) and interactive
   (#slippanel) views; each keeps only its own padding delta below. The slip
   background matches the page paper (not the lighter --card) so it blends in
   rather than reading as a white box; its border + shadow still delineate it.
   Every inner region inherits this one fill (see the .prose reset below). */
.slip, #slippanel {
  background:var(--paper); border:1.5px solid var(--ink); border-radius:4px;
  box-shadow:5px 5px 0 rgba(33,28,21,.10);
  font-family:'JetBrains Mono',monospace; color:var(--ink); min-height:420px;
}
.slip { padding:0 20px 18px; }
.slip__head {
  display:flex; justify-content:space-between; align-items:baseline;
  padding:18px 0 12px;
}
.slip__title {
  font-family:'Fraunces',serif; font-weight:600; font-size:21px;
}
.slip__meta {
  font-size:10.5px; text-transform:uppercase; letter-spacing:.14em;
  color:var(--ink-soft);
}
.slip__perf {
  border-top:2px dashed var(--line); height:0; margin:2px -20px;
}
.slip__body { padding:14px 0; }

.grp { margin-bottom:18px; }
.grp__name {
  font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.18em;
  color:var(--olive); padding-bottom:8px; margin-bottom:10px;
  border-bottom:1px solid var(--line);
}
.ln {
  display:grid; grid-template-columns:auto 1fr auto; gap:12px;
  align-items:start; padding:9px 0;
}
.ln + .ln { border-top:1px dotted var(--line); }
.ln__qty {
  font-weight:700; font-size:18px; min-width:34px;
}
.ln__x { color:var(--ink-soft); font-size:12px; margin-left:2px; }
.ln__name {
  font-family:'Fraunces',serif; font-size:16px; font-weight:500;
}
.ln__sub {
  font-size:11px; color:var(--ink-soft); margin-top:3px; letter-spacing:.04em;
}
.ln__price { font-size:13px; white-space:nowrap; padding-top:2px; }
.ln__flags { margin-top:7px; display:flex; flex-wrap:wrap; gap:6px; }
.chip {
  font-size:9.5px; text-transform:uppercase; letter-spacing:.1em;
  padding:2px 7px; border-radius:3px; border:1px solid;
}
.chip--warn { color:var(--stamp); border-color:var(--stamp); background:rgba(192,67,43,.07); }
.chip--info { color:var(--ink-soft); border-color:var(--line); }

.slip__foot {
  display:flex; justify-content:space-between; align-items:baseline;
  padding-top:14px; font-weight:700;
}
.slip__foot-label { font-size:11px; text-transform:uppercase; letter-spacing:.14em; }
.slip__foot-val { font-family:'Fraunces',serif; font-size:20px; }
.slip__note {
  font-size:10px; color:var(--ink-soft); margin-top:6px; font-style:italic;
}

/* Empty slip */
.slip--empty {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:18px; text-align:center;
}
/* In the interactive panel the centering wrapper has no .slip min-height of its
   own, so give it one to vertically center the stamp within the card. */
#slippanel .slip--empty { min-height:380px; }
.stamp {
  margin-top:60px;
  font-family:'JetBrains Mono',monospace; font-weight:700; font-size:22px;
  text-transform:uppercase; letter-spacing:.16em; color:var(--stamp);
  border:3px solid var(--stamp); border-radius:6px; padding:12px 22px;
  transform:rotate(-5deg); opacity:.55;
}
.slip__hint { color:var(--ink-soft); font-size:13px; line-height:1.7; max-width:280px; }

/* Interactive cart panel — the slip card wrapped around live Gradio components.
   Flatten Gradio's block/form wrappers so the components sit on the slip cleanly. */
#slippanel { padding:4px 20px 18px; gap:0 !important; }
#slippanel .block, #slippanel .form, #slippanel .html-container {
  background:transparent !important; border:none !important;
  box-shadow:none !important; padding:0 !important; overflow:visible !important;
}
/* Gradio's markdown wrapper (.prose) ships its own page-tinted fill; clear it so
   every region (group header, line item, subtotal) shows the one slip background
   instead of a faintly different box. */
#slippanel .prose { background:transparent !important; }
#slippanel > .gap, #slippanel .gap { gap:0 !important; }
.lnrow {
  align-items:center !important; gap:7px !important; flex-wrap:nowrap !important;
  padding:9px 0 !important; border-top:1px dotted var(--line);
  min-height:0 !important;
}
.lnrow .lncell { flex:1 1 auto; min-width:0; }

/* −/qty/+ as one connected pill (Amazon-style): a single rounded outline with
   flush, borderless buttons and the quantity centered between hairline dividers.
   The stepper hugs its content (flex:0 0 auto) so .lncell takes the rest of the
   row; ✕ stays a separate square control below. */
#slippanel .stepper {
  flex:0 0 auto !important; display:inline-flex !important; align-items:stretch !important;
  gap:0 !important; width:auto !important; min-width:0 !important;
  border:1.5px solid var(--ink); border-radius:6px; overflow:hidden;
  background:var(--paper-2);
}
#slippanel .stepper .block,
#slippanel .stepper .html-container,
#slippanel .stepper .qcell {
  flex:0 0 auto !important; width:auto !important; min-width:0 !important;
}
.stepper .qval {
  display:flex; align-items:center; justify-content:center; align-self:stretch;
  min-width:30px; padding:0 8px; text-align:center; font-weight:700; font-size:16px;
  border-left:1.5px solid var(--line); border-right:1.5px solid var(--line);
}
button.qbtn {
  flex:0 0 auto !important;
  width:32px !important; min-width:32px !important; max-width:32px !important;
  height:30px !important; padding:0 !important;
  border:none !important; border-radius:0 !important; box-shadow:none !important;
  background:var(--paper-2) !important; color:var(--ink) !important;
  font-family:'JetBrains Mono',monospace !important;
  font-size:18px !important; font-weight:700 !important; line-height:1 !important;
}
button.qbtn:hover:not([disabled]) { background:var(--paper) !important; }
button.rmbtn {
  flex:0 0 auto !important;
  width:30px !important; min-width:30px !important; max-width:30px !important;
  height:30px !important; padding:0 !important;
  border-radius:4px !important; box-shadow:none !important;
  font-family:'JetBrains Mono',monospace !important; line-height:1 !important;
  background:transparent !important; color:var(--ink-soft) !important;
  border:1.5px solid var(--line) !important; font-size:13px !important;
}
button.rmbtn:hover:not([disabled]) {
  background:var(--stamp) !important; color:#fbf7ee !important;
  border-color:var(--stamp-deep) !important;
}
""".replace("__GRAIN__", _GRAIN)


# gr.Button exposes no aria_label in Gradio 6, and @gr.render rebuilds the cart's
# buttons on every change, so a one-shot label wouldn't survive. Instead, on page
# load, attach a MutationObserver to the slip panel that (re)labels the glyph
# buttons by their text as rows are rendered — accessible names for screen readers
# without fighting the component API.
_A11Y_JS = """
() => {
  const panel = document.getElementById('slippanel');
  if (!panel) return;
  const LABELS = {'−': 'decrease quantity', '+': 'increase quantity', '✕': 'remove item'};
  const label = () => panel.querySelectorAll('button.qbtn, button.rmbtn').forEach((b) => {
    const name = LABELS[(b.textContent || '').trim()];
    if (name) b.setAttribute('aria-label', name);
  });
  label();
  new MutationObserver(label).observe(panel, { childList: true, subtree: true });
}
"""