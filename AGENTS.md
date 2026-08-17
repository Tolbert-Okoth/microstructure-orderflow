# Project Rules & Development Mandates: Quantitative XAUUSD Scalping Engine

## 🛑 MANDATORY CORE PRINCIPLES

### RULE 1: STRICTLY ZERO ASSUMPTIONS (DATA-GROUNDED EXECUTION)
- **Use Only Real, Local Data**: All analysis, model features, backtests, and calibrations MUST be derived strictly from the actual dataset we possess (e.g., MT5 tick/OHLC exports, empirical reference files, and downloaded verified literature in `data/`).
- **Never Rely on Model Training Biases or Hallucinated Stats**: Do not assume broker spreads, tick intervals, volatility parameters, or historical distributions from pre-trained memory. Every parameter must be explicitly computed or loaded from the concrete data files.
- **Verify All Inputs**: Validate data integrity (missing ticks, timestamp gaps, bid-ask spread anomalies) before passing data into any pipeline.

---

### RULE 2: NO SHORTCUTS (PRODUCTION-GRADE RIGOR)
- **Complete, Non-Trivial Implementations**: Absolutely no mock classes, placeholder methods (`pass` / `TODO`), hardcoded dummy signals, or toy approximations for critical mathematical models (Wasserstein HMM, TimesNet, Hurst Exponent, Ornstein-Uhlenbeck SDEs, PPO RL, or Friction-Adjusted Kelly Sizing).
- **Comprehensive Error Handling & Logging**: Every pipeline stage (data ingestion, feature engineering, regime classification, inference, execution, and sizing) must be fully engineered, typed, modularized, and validated.
- **Strict MT5 Protocol Adherence**: Maintain 100% compatibility with MetaTrader 5 data schemas and API structures.

---

### RULE 3: ZERO ROOM FOR ERROR IN BACKTESTERS & OPTUNA OPTIMIZATIONS
- **Absolute Exclusion of Look-Ahead Bias & Data Leakage**:
  - No future data leakage in scaling, normalization, feature creation, or regime segmentation.
  - Normalization parameters (e.g., mean, variance, Hurst rolling windows, HMM emissions) must be calculated strictly on causal, expanding or rolling historical windows.
  - Triple-Barrier labeling and walk-forward cross-validation must strictly adhere to causal time boundaries.
- **Realistic Microstructure & Execution Simulation**:
  - Zero unrealistic instant fills at candle extremes.
  - Strictly simulate dynamic spreads, slippage, intrabar wick penetration, overnight swap costs, and broker execution latency.
  - Dynamic trailing stops (e.g., 50-EMA on candle close) must be evaluated strictly at candle close timestamps, not retroactively within intrabar wicks.
- **Optuna & Optimization Integrity**:
  - Optimization objectives in Optuna must NOT optimize for raw, overfitted cumulative PnL. They must optimize for friction-adjusted, penalty-weighted risk metrics (e.g., Deflated Sharpe Ratio, Sortino Ratio, Calmar Ratio, penalized for excessive turnover and drawdown).
  - Use Purged & Embargoed Walk-Forward Cross-Validation or Combinatorial Purged Cross-Validation (CPCV) to guard against data snooping.
  - Enforce strict out-of-sample and out-of-distribution validation before any parameter set is accepted.

---

### RULE 4: DEEP CONTEXT EXTRACTION & EXHAUSTIVE RESEARCH GROUNDING
- **No Surface-Level Sampling or Cherry-Picking**: When reading, analyzing, or adapting methodologies from research papers, preprints, specifications, and PDFs in `data/`, do NOT take superficial snippets, isolated formulas, or partial notes without synthesizing the full context of the paper.
- **Full Theoretical & Mathematical Fidelity**: Ingest and implement the full mathematical framework—including all boundary conditions, microstructure noise filters, calibration algorithms, cost-drag models, execution constraints, and edge-case handling.
- **Go Deep and Finish**: Never stop halfway or substitute a complex derivation with a simplified heuristic. Trace every equation to its fundamental implementation, ensure full parameter convergence, and complete the full end-to-end quantitative pipeline.
