# Phase 1 Experiment Execution Guide

## Quick Start Commands
```bash
# 1. Activate venv
conda activate tracegen

# 2. Install Python dependencies
pip install -r tools/requirements.txt

# 3. Run pre-experiment checklist
./tools/pre_experiment_checklist.sh

# 4. Run a single experiment
python3 tools/run_single_experiment.py resnet50 1

# 5. Track progress
./tools/experiment_tracker.sh
```

## Experiment Matrix (9 Total)

| Workload | Replicas | Command |
|----------|----------|---------|
| resnet50 | 1 | `python3 tools/run_single_experiment.py resnet50 1` |
| resnet50 | 3 | `python3 tools/run_single_experiment.py resnet50 3` |
| resnet50 | 8 | `python3 tools/run_single_experiment.py resnet50 8` |
| distilbert | 1 | `python3 tools/run_single_experiment.py distilbert 1` |
| distilbert | 3 | `python3 tools/run_single_experiment.py distilbert 3` |
| distilbert | 8 | `python3 tools/run_single_experiment.py distilbert 8` |
| whisper | 1 | `python3 tools/run_single_experiment.py whisper 1` |
| whisper | 3 | `python3 tools/run_single_experiment.py whisper 3` |
| whisper | 8 | `python3 tools/run_single_experiment.py whisper 8` |

## Configuration

- **Scrape interval:** 5 seconds (720 data points/hour)
- **Startup delay:** 5 minutes (ensures steady-state)
- **Recording duration:** 60 minutes
- **Cleanup delay:** 30 seconds

## Data Output

CSV files will be saved to: `data/raw/phase1/`

Each experiment generates 8 CSV files:
- cpu_usage
- memory_usage
- gpu_utilization
- gpu_memory
- gpu_power
- cpu_psi
- memory_psi
- io_psi

Plus a timestamps file for reference.
