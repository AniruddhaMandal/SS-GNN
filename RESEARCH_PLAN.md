# SS-GNN Research Action Plan

**Date Created**: 2025-11-16
**Status**: Active Planning Phase

---

## Problem Statement

### Current Situation
- **Theoretical Claim**: There exists some k for which we can distinguish non-isomorphic graphs through the divergence of empirical distributions of k induced connected subgraphs
- **Implementation**: SS-GNN samples m many k-graphlets, encodes them with GNN, and aggregates with attention (temperature-based)
- **Issue**: Vanilla GNN outperforms SS-GNN on standard benchmarks (ZINC, PROTEINS, MUTAG)

### Performance Gap (Current Results)
| Dataset | Vanilla GNN (Test) | SS-GNN (Test) | Gap |
|---------|-------------------|---------------|-----|
| PROTEINS (GIN) | 70.5% | 71.4% | +0.9% (slight improvement) |
| MUTAG (GCN) | 73.7% | 63.2% | -10.5% (significant drop) |

### The Paradox
- UGS sampler: Lower coverage, higher CV → **Better performance**
- RWR sampler: Higher coverage, lower CV → **Worse performance**
- This suggests uniformity might NOT be the primary issue

### Current Implementation Details
- **Samplers**: UGS, RWR, uniform, epsilon_uniform (all implemented in C++)
- **Architecture**: SubgraphGNNEncoder + AttentionAggregator
- **Aggregator**: Attention with temperature parameter (current: 0.5)
- **Typical hyperparameters**: k=6, m=100 for PROTEINS
- **Location**: `src/gps/gps/models/ss_gnn.py`

---

## Research Plan Overview

Four parallel tracks with clear priorities and timelines:

1. **Track 1**: Validate Theory on Synthetic Data (HIGHEST PRIORITY - Week 1)
2. **Track 2**: Diagnose Real-World Performance (Week 2)
3. **Track 3**: Architectural Improvements (Week 3-4)
4. **Track 4**: Paper Positioning (Ongoing)

---

## Track 1: Validate Theory on Controlled Synthetic Data

**Priority**: HIGHEST
**Timeline**: Week 1
**Goal**: Determine if theory holds in practice when critical substructures are known to exist

### Phase 1.1: Create Synthetic Benchmark Datasets

**Objective**: Generate pairs of non-isomorphic graphs that vanilla GNN cannot distinguish

**Specific Datasets to Create**:

1. **Regular Graph Pairs**
   - Graphs with same degree sequence but different eigenspectra
   - Example: 3-regular graphs with different cycle structures

2. **WL-Test Failures**
   - Classic examples where WL color refinement fails
   - CFI graphs, Strongly Regular Graphs

3. **Triangle Counting Problems**
   - Graphs with identical node degrees but different triangle distributions
   - Critical substructure: k=3 triangles

4. **Clique Detection**
   - Graphs with different k-clique counts but same local neighborhoods
   - Critical substructure: k=4,5 cliques

5. **Tree-Width Variations**
   - Graphs with same basic statistics but different tree-width
   - Requires larger k to distinguish

**Deliverables**:
- 3-5 synthetic datasets, each with 1000+ graph pairs
- Ground truth labels (class 0 vs class 1 for graph pairs)
- Documentation of which k-substructures are critical for each dataset

**Success Criteria**:
- Vanilla GNN achieves ~50% accuracy (random guessing)
- Theoretically, SS-GNN with appropriate k should achieve >80% accuracy

### Phase 1.2: Systematic Testing on Synthetic Data

**Experimental Design**:
```
For each synthetic dataset:
  For k in [3, 4, 5, 6, 7]:
    For m in [50, 100, 200, 500]:
      - Train SS-GNN with UGS sampler
      - Train SS-GNN with RWR sampler
      - Train vanilla GNN baseline
      - Record test accuracy
```

**Metrics to Track**:
- Test accuracy (primary)
- Training convergence speed
- Attention weight entropy (are diverse subgraphs being used?)
- Coverage percentage (from profiling tools)

**Expected Outcomes**:

**Scenario A - Theory Validated**:
- SS-GNN significantly outperforms vanilla GNN (>20% gap)
- Performance correlates with appropriate choice of k
- Conclusion: Theory is sound, real-world datasets lack critical substructures

**Scenario B - Theory Not Validated**:
- SS-GNN fails to outperform even on synthetic data
- Indicates implementation issues or theory-practice gap
- Next step: Deep debugging (Track 2.1)

**Scenario C - Mixed Results**:
- Works for some synthetic datasets but not others
- Indicates sensitivity to substructure type
- Next step: Characterize when it works

---

## Track 2: Diagnose Real-World Dataset Performance

**Priority**: MEDIUM
**Timeline**: Week 2
**Dependency**: Start after Track 1 results

### Phase 2.1: Implementation Debugging

**Gradient Flow Analysis**:
```python
# Add hooks to track:
1. Gradient magnitudes at each layer
2. Dead neurons in encoder
3. Attention weight collapse (all weights → 1/m)
4. Subgraph embedding diversity (check if all embeddings are similar)
```

**Attention Mechanism Diagnostics**:
- Log attention weight distributions per graph
- Check for collapse: `max(attention_weights) > 0.9` indicates collapse
- Visualize which subgraphs receive high attention
- Verify temperature scaling is working correctly

**Encoder Verification**:
- Check subgraph embeddings are diverse (measure pairwise distances)
- Verify edge features are being used (for GINE)
- Test if encoder can distinguish different subgraph types

**Numerical Stability**:
- Check for NaN/Inf in attention scores
- Verify temperature parameter isn't causing overflow
- Test gradient clipping needs

**Implementation Location**: `src/gps/gps/models/ss_gnn.py:246-423`

### Phase 2.2: Systematic Hyperparameter Search

**Search Space**:
```yaml
k: [3, 4, 5, 6, 7, 8]
m: [20, 50, 100, 200, 500]
temperature: [0.1, 0.5, 1.0, 2.0]
aggregator: [attention, mean, max, sum]
sampler: [ugs, rwr]
learning_rate: [0.001, 0.0005, 0.0001]
dropout: [0.1, 0.3, 0.5, 0.6]
hidden_dim: [64, 128, 256]
num_layers: [3, 4, 5]
```

**Strategy**:
1. Use existing wandb sweep infrastructure (seen in `tune/wandb/sweep-*`)
2. Start with grid search on (k, m) - most critical
3. Then Bayesian optimization on other hyperparameters
4. Budget: ~200 runs per dataset

**Datasets to Focus On**:
- MUTAG (smallest, fastest iteration)
- PROTEINS (medium size)
- ZINC (if time permits)

**Current Config Location**: `configs/ss_gnn/TUData/gin-proteins.json`

### Phase 2.3: Coverage-Performance Correlation Analysis

**Use Existing Tool**: `tools/profile_sampler_cv.py`

**Analysis to Perform**:
```python
For each (dataset, k, m, sampler) configuration:
  1. Measure coverage percentage
  2. Measure coefficient of variation (CV)
  3. Calculate KL divergence from uniform distribution
  4. Train model and measure test accuracy
  5. Plot: coverage vs accuracy
  6. Plot: CV vs accuracy
  7. Plot: KL divergence vs accuracy
```

**Research Questions**:
- Does higher coverage → better accuracy?
- Does lower CV (more uniform) → better accuracy?
- Why does UGS (higher CV) beat RWR (lower CV)?

**Deliverable**: Analysis notebook with plots and correlations

---

## Track 3: Architectural Improvements

**Priority**: LOWER
**Timeline**: Week 3-4
**Dependency**: Start after understanding Track 1-2 results

### Phase 3.1: Alternative Aggregators

**Current**: Attention aggregator in `src/gps/gps/aggregator.py`

**New Aggregators to Implement**:

1. **Set2Set Aggregator**
   ```python
   # LSTM-based permutation invariant aggregation
   # Better at capturing multiset structure
   ```

2. **Deep Sets with Learnable Pooling**
   ```python
   # phi(x) for each subgraph, then rho(sum(phi(x)))
   # Provably universal for set functions
   ```

3. **Graph Multiset Transformer**
   ```python
   # Full self-attention over subgraph embeddings
   # Add positional encodings for subgraph order
   ```

4. **Weighted Mean Aggregator**
   ```python
   # Learn scalar weight per subgraph based on size/connectivity
   # Simpler than full attention
   ```

**Testing Protocol**:
- Keep encoder fixed, swap only aggregator
- Compare on best (k, m) from Track 2
- Measure: accuracy, training time, parameter count

### Phase 3.2: True Uniform Sampler Implementation

**Motivation**: Test if perfect uniformity helps

**Approach**:
```cpp
// Rejection sampling for small graphs (n < 100)
// 1. Enumerate all k-graphlets (use existing BFS code)
// 2. Sample uniformly at random from enumeration
// 3. Profile time cost vs UGS
```

**Implementation Location**: `src/samplers/uniform_sampler/`

**Constraints**:
- Only practical for small graphs
- Use as diagnostic tool, not production sampler
- Compare to UGS on same small graphs

**Expected Outcome**:
- If uniform helps significantly: UGS approximation quality matters
- If uniform doesn't help: Uniformity is not the bottleneck

### Phase 3.3: Add Subgraph-Level Features

**Hypothesis**: Model needs more context about each subgraph

**Features to Add**:
```python
For each sampled subgraph:
  1. Number of nodes (actual size, handle padding)
  2. Number of edges
  3. Average clustering coefficient
  4. Subgraph density
  5. Centrality of nodes in subgraph relative to full graph
  6. Subgraph "type" (e.g., path, cycle, tree, clique)
```

**Architecture Modification**:
```python
# Concatenate structural features to subgraph embedding
subgraph_emb = encoder(subgraph)  # [m, hidden_dim]
structural_features = compute_features(subgraph)  # [m, feat_dim]
combined = concat([subgraph_emb, structural_features], dim=-1)
graph_emb = aggregator(combined)
```

**Location to Modify**: `src/gps/gps/models/ss_gnn.py:378-423`

---

## Track 4: Paper Positioning Strategy

**Timeline**: Ongoing, finalize Week 4

### Option A: Theory Validated on Synthetic (Best Case)

**Title**: "Subgraph Sampling for Expressive Graph Neural Networks: Theory and Practice"

**Story**:
1. Theoretical contribution: Proof that k-subgraph distributions distinguish non-isomorphic graphs
2. Empirical validation on synthetic benchmarks
3. Analysis: Real-world datasets lack the critical substructures our theory requires
4. Contribution: Characterization of when subgraph sampling helps

**Sections**:
- Theory: Formal proof of expressiveness
- Synthetic Experiments: Validation on controlled data
- Real-World Analysis: Coverage profiling, substructure analysis
- Discussion: When to use subgraph sampling

### Option B: Implementation Insights (Fallback)

**Title**: "Practical Challenges in Subgraph-Based Graph Neural Networks"

**Story**:
1. Theoretical motivation for subgraph sampling
2. Implementation of multiple sampling strategies
3. Comprehensive empirical analysis reveals challenges
4. Insights for future work on practical subgraph GNNs

**Sections**:
- Related Work: Survey of subgraph GNNs
- Implementation: Detailed description of architecture choices
- Empirical Study: Extensive experiments with analysis
- Lessons Learned: What works, what doesn't, why

### Option C: Method Paper with Specific Contribution (Alternative)

**Title**: "SS-GNN: Attention-Based Aggregation for Subgraph Sampling"

**Story**:
1. Novel architecture: Attention aggregator for subgraph embeddings
2. Works well on specific datasets/tasks
3. Analysis of when it helps
4. Contribution: New aggregation mechanism

**Focus**:
- Architectural contribution (attention with temperature)
- Ablation studies on aggregator design
- Specific use cases where it excels

---

## Recommended Execution Order

### Week 1: CRITICAL PATH
**Goal**: Validate or refute theory

1. **Day 1-2**: Implement synthetic dataset generators
   - Start with triangle counting (simplest)
   - Then regular graph pairs
   - Then WL failures

2. **Day 3-4**: Run experiments on synthetic data
   - Sweep k ∈ {3,4,5,6} and m ∈ {50,100,200}
   - Compare vanilla vs SS-GNN
   - Analyze attention weights

3. **Day 5**: Analyze results and decide direction
   - If successful → Track 2.3 + Track 3
   - If unsuccessful → Track 2.1 (deep debugging)

### Week 2: DIAGNOSIS
**Goal**: Understand real-world performance

**Scenario A - Synthetic Succeeded**:
- Track 2.3: Coverage analysis on real datasets
- Hypothesis: PROTEINS/MUTAG lack critical substructures
- Analysis: What substructures exist? Do they matter for classification?

**Scenario B - Synthetic Failed**:
- Track 2.1: Implementation debugging
- Track 2.2: Hyperparameter search (maybe wrong config)
- Deep dive into attention mechanism

### Week 3-4: IMPROVEMENT or PAPER WRITING
- If making progress: Track 3 (architectural improvements)
- If clear story emerges: Write paper
- If stuck: Pivot to lessons learned paper

---

## Tools and Resources Available

### Existing Codebase Assets
- **Profiling Tools**:
  - `tools/profile_sampler_cv.py` - CV and coverage analysis
  - `tools/profile_sampler_dataset.py` - Dataset-wide profiling
  - `tools/profile_sampler_validity.py` - Sampler correctness

- **Samplers**:
  - `src/samplers/ugs_sampler/` - Unbiased subgraph sampler
  - `src/samplers/rwr_sampler/` - Random walk with restart
  - `src/samplers/uniform_sampler/` - Uniform sampler (baseline)
  - `src/samplers/epsilon_uniform_sampler/` - Approximate uniform

- **Experiment Infrastructure**:
  - Wandb integration for tracking
  - Config-based experiments
  - Multi-seed averaging
  - `gps-run` CLI tool

### Metrics to Track
1. **Performance Metrics**:
   - Test accuracy (primary)
   - Train/val accuracy (overfitting check)
   - AUC (for binary classification)

2. **Sampling Metrics**:
   - Coverage percentage
   - Coefficient of variation (CV)
   - KL divergence from uniform
   - Sampling time

3. **Model Diagnostics**:
   - Attention weight entropy
   - Gradient norms
   - Embedding diversity
   - Training convergence

---

## Decision Points and Contingencies

### Decision Point 1 (End of Week 1)
**Question**: Does SS-GNN work on synthetic data?

- **YES → Path A**: Theory is sound
  - Continue to real-world analysis
  - Focus on dataset characterization
  - Paper: Strong theoretical contribution

- **NO → Path B**: Implementation issues or theory gap
  - Deep debugging phase
  - May need to revise theory or architecture
  - Paper: Lessons learned / negative results

### Decision Point 2 (End of Week 2)
**Question**: Do we understand why real-world performance differs?

- **YES → Path A**: Clear story
  - Write paper with clear narrative
  - Highlight when method works

- **NO → Path B**: Continue investigation
  - More experiments needed
  - Consider architectural changes (Track 3)

### Contingency Plans

**If completely stuck (end of Week 2)**:
1. Pivot to purely empirical study
2. Focus on sampler comparison (UGS vs RWR vs uniform)
3. Contribution: Comprehensive benchmark of subgraph sampling methods

**If showing promise but not publishable (end of Week 3)**:
1. Target workshop instead of main conference
2. Position as work-in-progress
3. Emphasize open problems and future directions

---

## Success Metrics

### Minimum Viable Paper (Workshop Quality)
- [ ] Clear theoretical motivation
- [ ] At least one setting where SS-GNN outperforms baseline
- [ ] Thorough empirical analysis
- [ ] Honest discussion of limitations

### Strong Paper (Conference Quality)
- [ ] Novel theoretical contribution (expressiveness proof)
- [ ] Validation on synthetic benchmarks
- [ ] Analysis of real-world datasets
- [ ] Clear characterization of when method works
- [ ] Outperforms baselines in identified settings

### Exceptional Paper (Top Venue)
- [ ] All of the above, plus:
- [ ] Novel architecture contribution
- [ ] State-of-the-art on specific benchmarks
- [ ] Practical sampling algorithm with guarantees
- [ ] Open-source release with reproducible results

---

## Next Immediate Actions

### High Priority (Start Immediately)
1. **Implement synthetic dataset generator** for triangle counting
2. **Run baseline experiments** on synthetic data
3. **Set up diagnostic logging** for attention weights

### Medium Priority (Week 1-2)
1. **Design hyperparameter sweep** for MUTAG
2. **Run coverage profiling** on existing results
3. **Implement gradient flow tracking**

### Lower Priority (Week 2+)
1. **Try alternative aggregators** (Set2Set, etc.)
2. **Implement true uniform sampler** for small graphs
3. **Add structural features** to subgraph embeddings

---

## Open Questions to Investigate

1. **Why does UGS (higher CV) outperform RWR (lower CV)?**
   - Hypothesis: Diversity matters more than uniformity
   - Test: Measure embedding diversity directly

2. **What is the right k for each dataset?**
   - Hypothesis: k should match critical substructure size
   - Test: Analyze graph motifs in each dataset

3. **Is the attention aggregator learning useful patterns?**
   - Hypothesis: May be collapsing to uniform weights
   - Test: Log attention entropy, visualize high-attention subgraphs

4. **Do real-world datasets have distinguishing substructures?**
   - Hypothesis: PROTEINS/MUTAG may be solvable with local info
   - Test: Analyze if k-subgraph distributions actually differ between classes

5. **Is m=100 samples enough?**
   - Hypothesis: Need more samples for reliable empirical distribution
   - Test: Convergence analysis as m increases

---

## Contact and Collaboration

**Researcher**: Aniruddha Mandal
**Email**: ani96dh@gmail.com
**Repository**: https://github.com/AniruddhaMandal/SS-GNN

---

## Version History

- **v1.0** (2025-11-16): Initial research plan created
  - Four-track structure defined
  - Week 1 priority: Synthetic data validation
  - Decision tree for pivoting based on results

---

## Notes and Reflections

### Key Insights So Far
1. The UGS vs RWR paradox suggests uniformity may not be the key factor
2. Performance gap on MUTAG is severe (-10.5%) - needs explanation
3. Existing profiling tools are excellent - leverage them
4. Theory is sound, but practice may depend on dataset characteristics

### Risks and Concerns
1. **Risk**: Synthetic data may be too artificial
   - Mitigation: Use known GNN failure cases from literature

2. **Risk**: Hyperparameter search may be too expensive
   - Mitigation: Start with coarse grid, refine promising regions

3. **Risk**: Theory-practice gap may be fundamental
   - Mitigation: Position as characterization study

### References to Add
- Original UGS paper (Bressan M., 2020)
- WL-test expressiveness papers
- Subgraph GNN literature (GraphSAINT, etc.)
- Attention aggregation methods

---

**END OF RESEARCH PLAN**
