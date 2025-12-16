# AIC – High-Level Architecture

This document describes the high-level architecture of AI Incident Commander
(AIC). It intentionally avoids implementation details and algorithms.

---

## Architectural Overview

AIC is composed of four primary subsystems:

1. **Signal Ingestion Layer**
2. **Correlation Engine**
3. **Reasoning Layer**
4. **Interaction Layer**

All components are designed to run fully on-prem within customer infrastructure.

---

## 1. Signal Ingestion Layer

Responsible for ingesting and normalizing:
- Logs
- Metrics
- Deployment and change events

Key characteristics:
- Asynchronous ingestion
- Backpressure-aware
- Resilient to partial outages
- No dependency on external services

---

## 2. Correlation Engine

Responsible for:
- Temporal alignment of signals
- Identifying abnormal patterns
- Constructing incident context graphs

This layer performs **pre-LLM reasoning** to reduce noise and constrain inference.

---

## 3. Reasoning Layer

Uses a locally hosted language model to:
- Analyze correlated incident context
- Generate root-cause hypotheses
- Suggest investigative and remediation steps

The model does **not** receive raw production streams directly.

---

## 4. Interaction Layer

Interfaces with engineers via:
- APIs
- Dashboards
- Incident tooling integrations

Supports iterative, conversational investigation during active incidents.

---

## Trust Boundaries

- No outbound data transfer
- Explicit isolation between tenants
- Controlled model access
- Auditable inference actions
