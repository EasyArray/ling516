# Normalization / Simplification System Review (Summer Cleanup)

## Scope and context
This note captures a design-level review of the symbolic semantics workflow:

1. Students author lexicon entries in the DSL (backticks / `PhiValue`-producing expressions).
2. Instructors/students run an interpreter over syntax trees using composition rules.
3. The system displays normalized denotations and (sometimes) evaluated model values.

The goal is to separate concerns and reduce duplication while preserving the textbook-facing user experience.

## Current architecture (what works)

- `PhiValue` successfully acts as the core symbolic object (AST + optional type + display helpers).
- Simplification passes are modular and testable (`run_pass_tests()` path already exists).
- Interpreter/tree integration supports rich pedagogical display by showing intermediate denotations.

This foundation is good; most issues are boundary and policy drift, not core capability.

## Key design tension

Three responsibilities are currently intertwined:

- symbolic normalization (`AST -> AST`),
- model evaluation (`AST + env -> python value`),
- display policy (what to show, when, and where).

When these are mixed, behavior becomes implicit (e.g., “is this expression symbolic?” vs. “is this already interpreted?”), and DRY violations appear in name/guard handling.

## Recommended design decisions

### 1) Keep normalized expressions symbolic by default

- Preserve user-facing model notation (e.g., predicate names, domain constants).
- Inline only *symbolic aliases* that improve readability (e.g., local helper `PhiValue`s in rule code).
- Avoid inlining arbitrary Python runtime objects by default.

Proposed policy object:

- `NameResolutionPolicy.inline_phi_aliases = True`
- `NameResolutionPolicy.inline_literals = opt-in`
- `NameResolutionPolicy.inline_model_objects = False`

### 2) Separate value display from symbolic display

Use explicit display mode flags instead of ad hoc evaluation in rendering paths:

- `value_mode='off'`: show normalized denotations only.
- `value_mode='final'`: evaluate only final interpreter result.
- `value_mode='nodes'`: evaluate each node (advanced/debug mode).

This preserves pedagogical clarity for most exercises while enabling deeper model inspection when needed.

### 3) Unify guard logic in one module

Guard extraction, canonicalization, deduplication, and runtime semantics should live together.

Suggested module split:

- `p4s/core/guards.py`
  - guard AST shape helpers,
  - folding/unfolding utilities,
  - runtime lowering (`GuardMod -> IfExp`) hooks.

This reduces parallel implementations in simplifier passes and evaluator paths.

### 4) Centralize substitution and environment override semantics

Substitution currently appears across multiple components (inlining passes, beta reduction, call-time overrides).
Create a single internal service for substitution behavior and capture policy:

- capture-avoidance rules,
- keyword override semantics,
- name inlining policy integration.

### 5) Provide rule-pack presets for classroom workflows

Notebook workflows repeatedly define similar composition rule sets. Add built-in presets and let users override selectively:

- e.g., `Interpreter.preset('HK3')`, `preset('HK4')`, etc.
- or preconfigured subclasses with an ergonomic extension API.

This is likely the highest user-experience gain for the least conceptual overhead.

## Refactor roadmap (incremental)

1. Introduce policy objects (`NameResolutionPolicy`, `DisplayPolicy`) without changing defaults.
2. Move guard helpers into a shared module and rewrite existing users to call it.
3. Factor substitution behavior into one utility and route beta/inliner/call paths through it.
4. Add interpreter/node display modes and document recommended defaults for teaching.
5. Ship rule-pack presets; update notebooks to demonstrate extension rather than redefinition.

## Suggested acceptance checks

- Existing simplifier pass tests continue to pass.
- Notebook outputs remain symbolically stable under default settings.
- Node-value display can be toggled without changing symbolic denotations.
- No duplicated guard-folding logic remains outside shared guard utilities.

## Why this aligns with your stated goal

You asked for a system-level cleanup focused on normalization philosophy and student-facing use. This direction keeps the strongest part of the project—the readable, textbook-like denotation display—while making model evaluation and rendering choices explicit, configurable, and less duplicated in code.
