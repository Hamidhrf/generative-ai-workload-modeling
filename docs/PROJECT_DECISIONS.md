## Decision #2: Normalization + VM Config Conditioning

**Date**: January 23, 2026
**Phase**: Phase 2 - Data Preprocessing
**Status**: ✅ FINAL DECISION

### Decision
- Use StandardScaler normalization for training
- Store scaler parameters for denormalization  
- Include VM hardware config as model conditioning input

### Why
1. Better neural network training (normalized features)
2. Generate traces in absolute units (GB, cores, seconds)
3. Enable generation for different hardware configs
4. Support capacity planning and cost optimization

### See
- `VM_CONFIG_CONDITIONING.md` for complete strategy


Project Decision Log
Master's Thesis: Generative AI Workload Modeling
Author: Hamidreza Fathollahzadeh
Institution: Fachhochschule Dortmund
Date Started: December 2025
Purpose of This Document
This log documents critical architectural and methodological decisions made during the thesis project. It serves
as a reference to avoid repeating past mistakes and ensures consistency throughout the project.
Decision #1: Pod-Level Data Handling (DO NOT AGGREGATE)
Date: January 21, 2026
Phase: Phase 2 - Model Selection
Status: ✅ FINAL DECISION - DO NOT CHANGE
Decision
KEEP POD-LEVEL DATA SEPARATE - DO NOT AGGREGATE ACROSS PODS/REPLICAS
Context
During Phase 2 planning, there was initial confusion about whether to:
Option A: Aggregate metrics across pods (mean/sum) → 13 training samples
Option B: Keep each pod's trace separate → ~60 training samples
Reasoning
Thesis Goal (Line 62-64 of proposal):
"the model will support scalability by generating traces for each application scaled to 10×--100× replicas
at the pod level"
Why Pod-Level is Correct:
1. Thesis Requirement: Goal explicitly states "at the pod level" generation
2. More Training Data: 60 pod traces vs 13 aggregated experiments (4× more data)
3. Preserves Contention Effects: Individual pods experience different contention levels
4. Realistic Generation: Can generate diverse pod traces for r=50-100 scenarios
5. Research Focus: Project measures how contention affects individual pod behavior
6. Kwok Integration: Simulation environment likely needs individual pod traces
7. Experimental Design: Phase 1 measured individual pod behavior under contention
Why Aggregation is Wrong:
1. ❌ Loses pod-level variability (critical research data)
2. ❌ Reduces training data by 75%
3. ❌ Cannot generate individual pod traces for simulation
4. ❌ Doesn't match thesis objective
5. ❌ Throws away valuable contention measurements
Implementation
Data Structure:
python
# Each pod = 1 training sample # Each pod = 1 training sample
pod_traces pod_traces = = [ [
( (array array( (720 720, , 15 15) ), , { {'workload' 'workload': : 'resnet50' 'resnet50', , 'r' 'r': : 1 1, , 'pod' 'pod': : 'pod-1' 'pod-1'} }) ), ,
( (array array( (720 720, , 15 15) ), , { {'workload' 'workload': : 'resnet50' 'resnet50', , 'r' 'r': : 2 2, , 'pod' 'pod': : 'pod-1' 'pod-1'} }) ), ,
( (array array( (720 720, , 15 15) ), , { {'workload' 'workload': : 'resnet50' 'resnet50', , 'r' 'r': : 2 2, , 'pod' 'pod': : 'pod-2' 'pod-2'} }) ), ,
# ... ~60 total pod traces # ... ~60 total pod traces
] ]
Training Data Count:
ResNet50: r=1(1) + r=2(2) + r=3(3) + r=6(6) + r=10(10) = 22 pod traces
DistilBERT: r=1(1) + r=2(2) + r=6(6) + r=10(10) = 19 pod traces
Whisper: r=1(1) + r=2(2) + r=3(3) + r=5(5) + r=8(8) = 19 pod traces
Total: ~60 pod-level time series
Model Training:
Each pod trace is one training sample
Condition on replica count (r) during training
Model learns: "How does a pod behave when there are r total pods?"
Generation (r=50 scenario):
python
# Generate 50 individual pod traces # Generate 50 individual pod traces
for for pod_id pod_id in in range range( (50 50) ): :
trace trace = = model model. .generate generate( (condition condition= ={ {'r' 'r': : 50 50, , 'workload' 'workload': : 'resnet50' 'resnet50'} }) )
# trace shape: (720, 15) # trace shape: (720, 15)
Code Files Affected
✅ correct
data
_
_
loader.py - Loads pod-level traces
✅ phase2
_
eda.py - EDA on pod-level data
✅ lstm
_
baseline.py - Trains on pod-level traces
✅ All documentation files updated
References
Thesis proposal: Section 5 (Expected Outcome)
WHY
POD
_
_
LEVEL.md - Detailed explanation
correct
data
_
_
loader.py - Implementation
Important Note
⚠ DO NOT AGGREGATE PODS IN ANY FUTURE CODE
If someone suggests aggregation, refer back to this decision log and WHY
POD
LEVEL.md .
_
_
Decision #2: [Future Decision]
[Template for future decisions]
Decision Template
markdown
## ## Decision #N: [Title] Decision #N: [Title]
** **Date Date** **: [Date] : [Date]
** **Phase Phase** **: [Phase Number] : [Phase Number]
** **Status Status** **: [Draft / Final / Revised] : [Draft / Final / Revised]
### ### Decision Decision
[Clear statement of decision] [Clear statement of decision]
### ### Context Context
[Why this decision was needed] [Why this decision was needed]
### ### Reasoning Reasoning
[Why this decision was made] [Why this decision was made]
### ### Implementation Implementation
[How to implement this decision] [How to implement this decision]
Code Files Affected
### ### Code Files Affected
[List of files] [List of files]
### ### References References
[Related documents] [Related documents]
How to Use This Log
1. 2. 3. 4. Before making major decisions: Check if similar decision exists
When confused: Read relevant decision entry
When documenting: Reference decision number in code comments
When writing thesis: Use decisions to justify methodology
Last Updated: January 21, 2026



## Decision #3: EDA Findings Confirm Design Choices

**Date**: January 23, 2026
**Phase**: Phase 2 - EDA Complete
**Status**: ✅ Verified

### Key Findings

1. **Normalization Ranges Validated:**
   - All proposed absolute ranges appropriate
   - CPU max: 6.21 (range 0-8) ✓
   - Memory max: 3.36GB (range 0-10GB) ✓
   - Latency max: 1.69s (range 0-5s) ✓

2. **Contention Effects Confirmed:**
   - CPU usage increases with replica count
   - Latency degrades with contention
   - VM config conditioning is necessary

3. **Zero Metrics:**
   - io_psi and memory_psi are all zeros
   - No I/O or memory bottlenecks in experiments
   - Decision: Keep in dataset (15 metrics total)

4. **GPU Saturation:**
   - Mean utilization: 86.7%
   - Always >80% (saturated resource)
   - Confirms GPU is shared, contention-causing resource

### Impact on Model Design

- ✅ Absolute normalization strategy confirmed
- ✅ VM config conditioning justified
- ✅ 15 metrics (including zeros) is correct
- ✅ Pod-level approach validated