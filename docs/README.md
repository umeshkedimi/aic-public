# AI Incident Commander (AIC)

AI Incident Commander (AIC) is an on-prem, privacy-first AI system designed to
assist engineering teams during production incidents by correlating logs,
metrics, and deployment signals to generate root-cause hypotheses and
remediation guidance.

This repository intentionally contains **architecture, system design, and
engineering decisions only**. Core implementation and domain logic are
maintained in private repositories.

---

## Problem

Modern production incidents are complex, fast-moving, and noisy.

Existing observability tools surface signals:
- dashboards
- alerts
- traces

But during an outage, engineers struggle with the harder question:

**Why is this happening right now?**

Root-cause analysis is slow, manual, and experience-dependent — leading to
longer downtime and higher operational cost.

---

## Core Principles

- **On-prem / self-hosted**: no production data leaves customer infrastructure
- **Correlation > chat**: reasoning over signals, not generic Q&A
- **Human-in-the-loop**: assist engineers, never auto-remediate blindly
- **Deterministic + probabilistic reasoning**: combine rules with AI inference
- **Failure-aware design**: partial data, degraded systems, cascading outages

---

## Scope of This Repository

This public repository documents:

- System architecture
- Design tradeoffs
- MVP scope definition
- API contracts (high-level)
- Security and deployment considerations

Implementation details, prompts, heuristics, and tuning data are intentionally
not public.

---

## Status

AIC is under active MVP development.

Start here:
- ARCHITECTURE.md
- MVP_SCOPE.md
