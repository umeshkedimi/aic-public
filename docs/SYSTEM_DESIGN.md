# AIC – System Design Considerations

This document captures key system design decisions and tradeoffs in AI Incident
Commander (AIC). It focuses on reliability, scalability, and failure-aware
design rather than feature completeness.

---

## Design Goals

- Remain operational during partial system failures
- Minimize blast radius during overload
- Prioritize signal quality over volume
- Preserve deterministic behavior where possible
- Use AI for reasoning, not for ingestion or transport

---

## Ingestion Strategy

### Why Asynchronous Ingestion

Production incidents often coincide with traffic spikes and system stress.
Synchronous ingestion would amplify failures.

Asynchronous ingestion allows:
- Backpressure handling
- Graceful degradation
- Decoupling producers from consumers

---

## Data Normalization

Signals are normalized into a common incident context model:
- Time-bounded
- Source-tagged
- Confidence-scored

Normalization occurs **before** any AI reasoning.

---

## Correlation Before AI

Raw signals are too noisy for direct LLM consumption.

The correlation engine:
- Aligns signals temporally
- Filters irrelevant noise
- Builds a constrained incident context graph

This reduces hallucination risk and improves reasoning accuracy.

---

## LLM Usage Boundaries

The language model:
- Does not ingest raw streams
- Operates on curated incident snapshots
- Produces hypotheses, not decisions
- Is sandboxed from execution paths

This keeps the system predictable and auditable.

---

## Failure Modes

AIC is designed to handle:
- Partial data loss
- Delayed signals
- Conflicting indicators
- Model unavailability

In degraded mode:
- Deterministic heuristics remain active
- AI reasoning becomes advisory or disabled

---

## Scalability Model

- Horizontal scaling for ingestion
- Incident-scoped compute isolation
- Bounded memory per incident

AIC scales by **incident count**, not raw signal volume.

---

## Observability of AIC Itself

AIC instruments:
- Ingestion lag
- Correlation depth
- Hypothesis latency
- Model response quality

This ensures AIC does not become a blind spot during incidents.
