# Prediction Lab — Deep Olive Design System

## 1. Brand premise

**Prediction Lab** is positioned as a small applied-intelligence product: it helps people inspect model outputs, understand confidence, and compare scenarios.

The interface should feel:

- calm rather than flashy
- analytical rather than "AI-themed"
- professional without looking corporate
- distinctive without relying on novelty
- useful before it is decorative

The visual system exists to communicate **grounded applied intelligence**.

---

## 2. Core palette

### Brand

- Deep Olive 800 — `#303A29`
  - Primary brand rail and major identity surface.
- Deep Olive 700 — `#4B5A38`
  - Secondary brand action / hover.
- Olive 500 — `#718C4A`
  - Primary interactive accent and active state.
- Olive 400 — `#8EA463`
  - Secondary emphasis.
- Olive 100 — `#EDF1E7`
  - Soft brand surface.
- Olive 050 — `#F5F7F0`
  - Very light brand wash.

### Neutrals

- Ink 950 — `#22261F`
  - Primary text and headings.
- Ink 800 — `#353A31`
  - Secondary text and important labels.
- Ink 600 — `#60665B`
  - Body/supporting text.
- Ink 500 — `#7D8378`
  - Metadata and tertiary text.
- Ink 200 — `#D9DDD5`
  - Dividers and subtle borders.
- Canvas — `#F6F7F2`
  - Main application background.
- Surface — `#FFFFFF`
  - Cards and forms.

### Semantic

- Positive — `#4E7E58`
- Positive soft — `#EEF6EF`
- Risk — `#B9564B`
- Risk soft — `#FBEEEC`

Semantic colors communicate prediction outcomes only. They are not general brand colors.

---

## 3. Color rules

1. The dark olive rail establishes identity.
2. The main workspace remains mostly neutral.
3. Olive is used for interaction, active states, and product-level emphasis.
4. Green/red are reserved for prediction semantics.
5. Do not use blue as a default UI accent.
6. Do not fill entire screens with the brand color.
7. Avoid gradients unless they become necessary for a data visualization.

---

## 4. Typography

The default system sans-serif is intentional.

Hierarchy:

- Page title: 2.15rem, weight 780
- Result headline: 2.2–3.4rem, weight 800
- Section heading: 0.89rem, weight 780
- Body: 0.78–0.96rem
- Metadata: 0.68–0.74rem
- Kicker / eyebrow: 0.68rem, weight 800, letter spacing 0.12–0.14em

Do not use giant hero typography. The result number is the strongest visual element because it is the core output.

---

## 5. Layout

### Shell

- Persistent dark olive left rail
- Wide neutral workspace
- Maximum content width: approximately 1320px
- Generous horizontal padding
- 12-column mental grid

### Information hierarchy

The screen should read in this order:

**What am I looking at? → What is the prediction? → How confident is it? → What inputs produced it? → What should I try next?**

---

## 6. Primary components

### Result card

The most important component.

Contains:

- result label
- large prediction
- short interpretation
- optional confidence
- model metadata

It should never be hidden below a large input form once a prediction exists.

### Context card

Shows the input profile associated with the prediction.

Purpose: preserve traceability.

### Explanation card

Explains how to interpret the output and prevents users from over-reading the result.

Purpose: trust and responsible ML communication.

### Helper strip

A quiet one-line instruction near the page header.

Purpose: answer "what am I supposed to do here?"

### Scenario section

Allows the user to create another prediction and compare outcomes.

Purpose: turn prediction into exploration rather than a one-shot form submission.

---

## 7. UX copy principles

Avoid:

- "AI-powered"
- "revolutionary"
- "next-generation"
- "engine"
- "unlock"
- "magic"
- "insights" when the text does not actually provide one

Prefer:

- "Estimate"
- "Run prediction"
- "Current assessment"
- "Prediction profile"
- "Model confidence"
- "New scenario"
- "How to use the estimate"
- "How to read the result"

The interface should sound like a competent analyst, not a marketing landing page.

---

## 8. Responsible ML communication

Predictions should be described as predictions.

For housing:

> "This is a model estimate — not a formal appraisal."

For Titanic:

> "The output describes model behavior on the supplied profile — not historical certainty."

Do not present predictive features as causal drivers unless the backend actually provides validated explanatory analysis.

---

## 9. Brand voice

Use short, plain sentences.

The product should sound:

**calm, precise, evidence-aware, confident without overclaiming.**

---

## 10. Extension rules

Any future model should inherit the same shell.

Only these should change:

- model name
- model description
- input schema
- result semantics
- model-specific interpretation

The navigation rail, typography, spacing, cards, button language, and visual hierarchy should stay consistent.
