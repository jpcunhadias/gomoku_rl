# Documentation

This directory contains all project documentation organized by lifecycle stage.

## Structure

- **`CHANGELOG.md`** - Consolidated history of all training cycles
- **`current/`** - Active documentation for the current cycle being worked on
- **`archive/`** - Historical documentation from completed cycles

## Current status

Cycle 2 (v4) is done and validated — see `archive/CYCLE2_V4_VALIDATION_AND_ARENA.md` for the
full result (Cycle 2 beats Cycle 1: 100W-0L-100D) and `archive/CYCLE1_COLDSTART_MECHANISM.md`
for why v4 only worked once there was a trained model to search with. No cycle is currently
in progress — `current/` is empty. Open items for whoever picks this up next: the value-head
overconfidence flagged in Cycle 2's debug checks, the planned tau/dirichlet_epsilon sweep, and
DVC/MLflow backup wiring (see `CHANGELOG.md`'s "Next Steps").

## Workflow

1. **During Cycle Development**: Keep active docs in `current/`
2. **After Cycle Success**: Move docs from `current/` to `archive/` and update `CHANGELOG.md`
3. **Start New Cycle**: Create new docs in `current/` for the new cycle

## Archive

Historical cycle documentation preserved for reference:
- Cycle 1 exploration analysis
- Cycle 2 changes and validation reports
- Previous cycle training adjustments

