# Open-Core Boundary (decision doc)

**Status:** decision recorded, not yet acted on. There is currently **one public
repository** and **no paid tier**. This doc exists so that *when a new feature is
built, we know which side of the line it belongs on* — and so we don't
accidentally wall off something that drives adoption.

**Do not build a private repo or paid features on speculation.** The trigger for
actually splitting is a **real paying customer** whose need forces the boundary,
not a roadmap date. Until then, everything ships in the public source-available
core.

## Guiding principle

> Individuals and labs use Calibra **locally**, for free. Companies pay when
> Calibra becomes a **shared, persistent, supported** part of their production
> data workflow.

The public core must be enough for a lab to fully inspect, reproduce, and trust
the science. The paid layer is about *organizational* workflows — multi-user,
persistent, hosted, governed — not about gating the core capability.

## Public core (source-available, the adoption engine)

Keep these public. They are what earns trust and drives adoption; walling any of
them off would damage the wedge.

- Core metrics and analyzers
- Dataset adapters / format support
- The local CLI (all 14 commands, single-user)
- `audit`, `compare`, `corrupt`
- Baseline `certify`, `prune`, `predict`
- Benchmark scripts **and results** (the honesty/trust mechanism)
- Claim registry and reference profiles
- Local, single-user dashboard
- SDK and integration examples

## Paid layer (build later, customer-triggered)

These are organizational-workflow features — defer until a customer needs one.

- Team workspaces / multi-user
- Persistent org-wide history
- Centralized outcome database (the "Training History Database")
- Customer-specific predictor calibration
- Fleet and operator monitoring (hosted `watch` aggregation)
- Dataset lineage and versioning
- Enterprise dashboard
- Alerts and scheduled reports
- SSO, RBAC, audit logs
- On-prem / VPC management tooling
- Proprietary integrations and support tooling
- Cross-customer benchmark intelligence (permissioned/anonymized data only)

## Rules of thumb when a new feature appears

1. **Does a solo researcher need it to reproduce or trust the results?** → public.
2. **Is it single-user and local?** → public.
3. **Does it only make sense with multiple users, persistence, or hosting?** → paid layer.
4. **Would gating it make a lab distrust the benchmarks?** → public, always.
5. **Unsure?** → ship public. It's cheap to withhold the *next* feature; it's
   expensive (and trust-damaging) to claw back one people already depend on.

## Not now (explicitly deferred until a paying customer exists)

- Creating the private repository
- Building any paid feature above
- Drafting commercial-license terms beyond the existing contact line
- Engaging an IP lawyer to review the BSL grant / contributor ownership

The existing README contact line for commercial licensing is sufficient to field
an inbound request. The first real request is the signal to start this work.
