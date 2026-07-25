# RHC_Analytics

A right heart catheter (RHC) is an invasive procedure doctors use to closely monitor a critically ill patient's heart and circulation. It's common in ICUs, but doctors have long disagreed about whether it actually helps patients or does more harm than good. This project uses real data from 5,735 ICU patients to investigate that question, along with a few related ones, using statistical and machine learning techniques common in healthcare and business analytics.

## What this project asks

- **Does the procedure actually change a patient's chance of dying within 30 days?** Or does it just look that way because doctors tend to give it to patients who are already sicker?
- **What patient characteristics best predict who is at high risk of dying?**
- **Does the procedure help, hurt, or make no difference for specific types of patients**, for example, people with different underlying illnesses, or different levels of severity?
- **Does the procedure change how quickly someone dies**, not just whether they do? *(not yet answered, see Status below)*

## The data

The dataset covers 5,735 ICU patients: their health at admission, whether they received the procedure, and what happened over the next 30 days. Some fields were missing or incomplete; each was handled based on *why* it was likely missing rather than one blanket rule, for example, a missing "date of death" almost always just means the patient survived, not that data was lost.

## Does the procedure actually increase the risk of dying?

At first glance, yes: patients who received the procedure died within 30 days about 38% of the time, versus 31% for patients who didn't.

That comparison isn't fair on its own, though. Patients who received the procedure were also considerably sicker to begin with, worse vital signs, more severe illness scores, more organ dysfunction, so some or all of that 7-point gap could simply reflect how sick they already were, not any effect of the procedure itself.

To account for this, we built a statistical model estimating how likely each patient was to receive the procedure, based on everything known about them at admission. We used that model to reweight the patients so the "received the procedure" group and the "didn't" group become as comparable as possible on every measured characteristic, recreating a fair comparison out of two originally mismatched groups.

After that adjustment, the gap shrank but didn't disappear: patients who received the procedure were still roughly 5 percentage points more likely to die within 30 days than similar patients who didn't (most likely somewhere between about 2 and 9 points). Because that remaining gap can't be explained away by measurable differences in how sick patients were, it points toward the procedure carrying some real added risk.

One honest caveat: this is real-world (observational) data, not a controlled experiment, so some unmeasured factor could still be influencing both who received the procedure and who survived. This result can't prove the procedure *causes* worse outcomes, only that the two are linked even after accounting for everything measured, which mirrors a long-standing, still-unresolved debate in the medical literature over whether this procedure is safe.

## Who is most at risk of dying?

Separately, we built a model that predicts a patient's risk of dying within 30 days using only information available at admission (not whether they received the procedure). We tested a few different modeling approaches and kept whichever performed best on patients it hadn't seen before.

The best model correctly predicted survival or death for about 74% of new patients, meaningfully better than a coin flip, though far from perfect. The strongest predictors of risk were a pre-calculated survival-probability score already on file for each patient, a patient's baseline functional health, whether a do-not-resuscitate order was in place, and liver function.

## Does the effect differ for different types of patients?

We also checked whether the procedure's effect changes depending on a patient's underlying diagnosis (for example, heart failure vs. liver disease vs. a severe body-wide infection) or how severely ill they were on arrival.

We did not find strong evidence that the effect meaningfully differs across these patient groups. One severe-infection-related diagnosis looked somewhat worse on its own, but a more rigorous check comparing all groups directly did not confirm that was a real, reliable difference rather than chance. A couple of the smaller diagnosis groups didn't have enough patients to draw any conclusion from and were excluded rather than reported with a falsely precise number.

## Status

This project is still in progress (the timing question above isn't answered yet). Charts and full results tables live in the `figures/` and `csv/` folders. A more detailed technical write-up, covering the full methodology and statistics, will be added once complete.
