---
name: bayesian-workflow
description: >
  Bayesian modeling workflow and iterative model-building strategy. Use when planning a
  modeling approach, deciding how to build up from simple to complex models, choosing between
  model specifications, or discussing Bayesian workflow principles. Triggers on: Bayesian
  workflow, iterative modeling, model building strategy, model expansion, prior predictive
  simulation, fake data simulation, simulation-based calibration, model criticism, combining
  information from multiple sources, or when a user asks "how should I model this?"
---

# Bayesian Workflow

Principled approach to iterative Bayesian model building, drawing on Gelman, Vehtari, and McElreath (2025) and Gelman et al. (2020).

Real-world Bayesian modeling is not "specify a model, fit it, report results." It is an iterative process of building up models, checking them against data and domain knowledge, and expanding only when simpler models demonstrably fail. Every model exists in relation to other models.

For PyMC implementation details (API patterns, samplers, diagnostics code), see the `pymc-modeling` skill. This skill covers the *strategy* of how to build and evaluate models.

## The Iteration Loop

```
specify → simulate fake data → check priors → fit simple model → diagnose →
    → criticize with posterior predictive checks → expand if needed → compare → report
```

Do not skip steps. Each one catches different problems.

## Start Simple, Build Up

Begin with the simplest model that could plausibly address the question. Fit it. Understand it. Then add complexity one piece at a time.

**Why this matters:**
- Simpler models are easier to understand and debug
- Complex models are best understood in relation to simpler special cases
- Each expansion reveals what the added complexity buys (or doesn't)
- Computational problems often emerge from modeling problems — starting simple isolates them
- Sometimes the simple model is sufficient, and you can demonstrate this by showing the expansion doesn't help

**Example sequence for a treatment effect study:**
1. Complete pooling (single mean per group)
2. Add pre-treatment covariates one at a time — see what each adjustment does
3. Allow treatment effects to vary by subgroup (partial pooling)
4. Add nonlinearity if residual patterns demand it

At each step: diagnose convergence, check posterior predictive fit, compare to the previous model.

## Simulate Before You Fit

Before touching real data, simulate from your model with known parameter values. Then fit your model to the simulated data and check whether it recovers the true parameters.

This is not optional. It catches:
- Specification bugs (wrong indexing, shape mismatches)
- Non-identifiability (parameters the data cannot distinguish)
- Prior-likelihood conflict (priors too strong or too weak)
- Computational problems unrelated to real-data messiness

```
1. Choose plausible parameter values (informed by domain knowledge)
2. Simulate a dataset from the generative model
3. Fit the model to simulated data
4. Check: are the true parameters recovered within posterior intervals?
5. Repeat with different parameter values and sample sizes
```

For a more systematic version, see simulation-based calibration (SBC): generate many fake datasets, fit each one, and check that posterior intervals have correct coverage.

**Perturbation experiments:** Vary the conditions — change sample size, move design points closer together, add outliers, try parameter values at the boundaries of plausibility. Understand where the fitting procedure breaks down.

## Prior Predictive Criticism

Prior predictive checking is not just "do the priors look reasonable." It is asking: **does my generative model produce datasets that look like plausible datasets from this domain?**

Push beyond summary statistics:
- Simulate full datasets from the prior predictive and plot them
- Check implied ranges on *observable* quantities, not just parameters
- Look for absurd implications (negative counts, probabilities outside [0,1], impossibly large effect sizes)
- If prior predictive datasets look nothing like your data, your priors are encoding wrong information — fix them before fitting

Prior predictive checks are especially important for complex models where the interaction between priors on multiple parameters creates unexpected behavior on the outcome scale.

## Model Criticism and Posterior Predictive Checks

After fitting, check whether the model captures the relevant structure of the data.

The goal is **not** to "accept" or "reject" the model — all models are wrong. The goal is to find *specific, interpretable ways* the model fails, which tell you what to fix.

- **Posterior predictive checks:** Simulate replicated datasets from the posterior. Do they resemble the observed data? Look at distributions, summary statistics, and patterns the model should capture.
- **Residual patterns:** Are there systematic deviations? These suggest missing structure (nonlinearity, heteroscedasticity, clustering).
- **Cross-validation:** LOO-CV identifies observations the model predicts poorly. High Pareto-k values point to influential observations or model misspecification.
- **Focus on the quantities that matter:** If you care about tail behavior, check tails. If you care about group differences, check group-level predictions. Generic goodness-of-fit tests are less useful than targeted checks.

## The Folk Theorem

> "When you have computational problems, often there's a problem with your model."
> — Andrew Gelman

When sampling is slow, divergent, or produces poor diagnostics:
- The problem is usually the model, not the sampler
- Reparameterize (non-centered for weak data, centered for strong data)
- Simplify — remove complexity until it works, then add back piece by piece
- Check for near-non-identifiability (highly correlated parameters, ridges in the posterior)
- **Do not** reach for sampler tuning knobs (more warmup, higher tree depth) as a first resort

Computational problems are diagnostic information about your model.

## Combining Information from Multiple Sources

Hierarchical models allow partial pooling across data sources without assuming they are identical or treating them as completely separate.

**When to partially pool:**
- Data from related experiments, populations, or time periods
- Information from different measurement instruments
- Prior information from published studies or expert knowledge

**The key principle:** You rarely want complete pooling (ignoring differences) or no pooling (ignoring similarities). Hierarchical structure lets the data determine how much to share.

Prior distributions are another way to incorporate external information — a prior on an effect size informed by previous studies is combining information, not "being subjective."

## Model Expansion as Understanding

Fit the expanded model even when you believe the simpler one is sufficient. The comparison itself is informative:

- If the expansion doesn't improve fit, you've demonstrated the simpler model is adequate — this is a finding worth reporting
- If it does improve fit, you've identified important structure
- The *difference* between models tells you something about the data-generating process

This applies to:
- Adding predictors to a regression
- Allowing parameters to vary by group
- Replacing a parametric assumption with something more flexible
- Adding a measurement error layer

## Measurement and Latent Variables

Real measurements are noisy proxies for the quantities we care about. When measurement error is non-trivial relative to the effects you're estimating:

- Add a measurement model layer that bridges observed data and latent quantities
- This is straightforward in the Bayesian framework: latent variables get priors and are estimated jointly with everything else
- Ignoring measurement error biases estimates (typically attenuation bias — effects look smaller than they are)

Common examples: test scores as proxies for ability, self-reported data, instruments with known precision.

## Reporting

Report the workflow, not just the final model:

- What simpler models did you try first? Why did you move beyond them?
- What did prior predictive checks reveal about your assumptions?
- What did posterior predictive checks reveal about model fit?
- What expansions did you try that didn't help?
- How sensitive are conclusions to reasonable alternative specifications?

The sequence of models *is* the analysis. The final model alone tells an incomplete story.

## References

- Gelman, Vehtari, and McElreath (2025). "Statistical Workflow." *Philosophical Transactions of the Royal Society A.*
- Gelman, Vehtari, Simpson, et al. (2020). "Bayesian Workflow." arXiv:2011.01808.
- Gabry, Simpson, Vehtari, Betancourt, and Gelman (2019). "Visualization in Bayesian workflow." *JRSS A.*
