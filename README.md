# RHC_Analytics
An analytics driven project that focuses on Right Heart Catherization (RHC) data. 

## Problem Statement

This project uses observational data from 5,735 ICU patients to answer the following questions:

1. **Does RHC cause a change in 30-day mortality risk (causal question)?** Right heart catheterization is an invasive procedure whose actual benefit vs. risk is disputed. Separating its true effect from the fact that sicker patients were more likely to receive it is the central question of this project.
2. **Which patient characteristics most predict mortality risk (predictive/risk-scoring question)?** Understanding what drives risk at admission is useful independent of any treatment decision, and demonstrates applied risk-scoring skills.
3. **Does the treatment effect differ across patient subgroups (segmentation question)?** An intervention that helps on average may not help, or may even harm, specific subgroups; averages alone can hide this.
4. **How does mortality risk evolve over the 30-day window, and does RHC affect timing, not just whether death occurs (survival/time-to-event question)?** Two patients who both survive or both die within 30 days can have very different trajectories; this question asks whether RHC shifts *when* death occurs, not just *whether* it occurs.
