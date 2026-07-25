# RHC_Analytics

A right heart catheter (RHC) is an invasive procedure doctors use to closely monitor a critically ill patient's heart and circulation. It's common in ICUs, but doctors have long disagreed about whether it actually helps patients or does more harm than good. This project uses real data from 5,735 ICU patients to investigate that question, along with a few related ones, using statistical and machine learning techniques common in healthcare and business analytics.

## What this project asks

- **Does the procedure actually change a patient's chance of dying within 30 days?** Or does it just look that way because doctors tend to give it to patients who are already sicker?
- **What patient characteristics best predict who is at high risk of dying?**
- **Does the procedure help, hurt, or make no difference for specific types of patients**, for example, people with different underlying illnesses, or different levels of severity?
- **Does the procedure change how quickly someone dies**, not just whether they do?

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

## Does the procedure change how quickly patients die?

Earlier sections ask *whether* a patient dies within 30 days. This section asks a related but different question: *when*. Two patients can both survive, or both die, within the 30-day window and still have very different paths, dying on day 2 is not the same as dying on day 28.

Tracking survival day by day, patients who received the procedure died sooner, on average, than patients who didn't; the gap between the two groups shows up within the first week and keeps widening through day 30. A formal statistical test confirms this difference is very unlikely to be due to chance.

That comparison is still unadjusted, though, the same "sicker patients were more likely to get the procedure" problem from earlier applies here too. Accounting for how sick each patient was at admission, patients who received the procedure faced roughly 25% higher risk of dying at any given moment during the 30 days than similar patients who didn't (most likely somewhere between about 12% and 38% higher). That's a separate confirmation, using a completely different method than earlier, of the same conclusion: the procedure is linked to worse outcomes even after accounting for how sick patients already were.

This approach also flags which factors are most strongly linked to a faster time to death, largely the same ones the earlier risk model found: a do-not-resuscitate order, a low pre-calculated survival-probability score, poor liver function, and being in a coma at admission. One technical caveat: for a handful of specific factors, their effect on risk changes noticeably over the 30 days rather than staying constant, but the procedure's own risk estimate held steady throughout the window, so that caveat doesn't weaken the finding above.

## Status

All four questions above are now answered. Charts and full results tables live in the `figures/` and `csv/` folders. A more detailed technical write-up, covering the full methodology and statistics, will be added once complete.
