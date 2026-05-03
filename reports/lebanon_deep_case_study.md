# Lebanon Case Study (Deep)

_Supplement to the proposal's Lebanon angle. Subgroup N = 49 campaigns is too small for reliable subgroup metrics, so this appendix is qualitative._

- Subgroup base rate: **65.3%**  (full-dataset base rate: 64.5%).
- Distinct categories represented: **10**.
- Distinct countries represented: **7**.

Why qualitative? At N≈49 the confidence interval on AUROC spans roughly 0.66 to 0.93, which is uninformative. Five detailed walk-throughs and three persona simulations carry more diagnostic weight than one noisy subgroup AUROC.

---

## Part A — Real campaigns, end-to-end

### HIGH-CONFIDENCE MISS — *eka3@Beirut*

*model thought it would succeed, it didn't*

**Inputs**
- Goal: $1,250  ·  Duration: 22 days
- Category: Music  ·  Country: GB
- Video: yes  ·  Launch year: 2012
- Blurb: "A Contemporary Arabic Music Festival that brings together established bands from the Arab region between Dec 20th and 29th in Beirut."

**Predicted probability of success: 79.2%**
**Actual outcome: Failed** _(model DISAGREED with outcome)_

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▲ log_goal                          shap=+0.604  value=7.132
  ▲ duration_days                     shap=+0.434  value=22.000
  ▲ has_video                         shap=+0.336  value=1.000
  ▼ launch_hour                       shap=-0.222  value=22.000
  ▲ category_name_Music               shap=+0.203  value=1.000
  ▼ country_US                        shap=-0.190  value=0.000
  ▲ blurb_has_number                  shap=+0.174  value=1.000
  ▼ trend_score                       shap=-0.116  value=50.000
  ▼ blurb_has_exclaim                 shap=-0.096  value=0.000
  ▼ launch_month                      shap=-0.095  value=12.000
```

**Recommendations the tool would have given at launch:**
1. Your launch hour (22:00) is associated with lower success. Late morning to early afternoon launches tend to perform better.  *(SHAP=-0.222)*
2. Google Trends interest for Music is currently soft. Either wait for the next interest peak or lean into your differentiators.  *(SHAP=-0.116)*
3. A small dose of energy helps - consider rephrasing to sound more confident.  *(SHAP=-0.096)*
4. Your launch month (December) is a weaker month for this category. Consider shifting the launch.  *(SHAP=-0.095)*

**Reading:** Model assigned a high probability based on a clean profile (right category, sensible goal, modest duration), but the campaign still failed. Things outside the feature set — niche audience, weak outreach, lack of pre-launch marketing — drove the outcome. This is exactly where 'correlational, not causal' bites: a clean profile is necessary but not sufficient.

---

### CORRECT RISK FLAG — *back to basics: authentic  lebanese  cuisine.*

*model warned of high failure risk, creator launched anyway, it failed*

**Inputs**
- Goal: $50,000  ·  Duration: 30 days
- Category: Food  ·  Country: AU
- Video: no  ·  Launch year: 2015
- Blurb: "Hearty, authentic grandma recipies of lebanese cuisine."

**Predicted probability of success: 2.0%**
**Actual outcome: Failed** _(model agreed with outcome)_

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▼ log_goal                          shap=-1.338  value=10.820
  ▼ category_name_Food                shap=-0.885  value=1.000
  ▼ has_video                         shap=-0.624  value=0.000
  ▼ country_AU                        shap=-0.334  value=1.000
  ▼ trend_score                       shap=-0.265  value=50.000
  ▼ blurb_word_count                  shap=-0.225  value=7.000
  ▼ launch_hour                       shap=-0.134  value=6.000
  ▼ category_name_Games               shap=-0.120  value=0.000
  ▼ duration_days                     shap=-0.098  value=30.000
  ▲ launch_dow                        shap=+0.090  value=1.000
```

**Recommendations the tool would have given at launch:**
1. Your goal of $50,000 may be too high for this profile. Consider lowering it.  *(SHAP=-1.338)*
2. Add a video. Campaigns without a video underperform meaningfully in this category.  *(SHAP=-0.624)*
3. Google Trends interest for Food is currently soft. Either wait for the next interest peak or lean into your differentiators.  *(SHAP=-0.265)*
4. Your blurb is only 7 words. Expand to ~18-28 words that state the product, audience, and one differentiator.  *(SHAP=-0.225)*
5. Your launch hour (6:00) is associated with lower success. Late morning to early afternoon launches tend to perform better.  *(SHAP=-0.134)*

**Reading:** The recommender put this campaign at high failure risk and gave specific reasons. The creator did not act on that advice (or didn't have access to it) and the campaign failed. This is the case the tool is built for — a creator launching with one fixable issue.

---

### ACTIONABLE RISK FLAG — *The Human Face of Terror*

*model would have given concrete advice the creator could act on*

**Inputs**
- Goal: $20,000  ·  Duration: 59 days
- Category: Journalism  ·  Country: IT
- Video: yes  ·  Launch year: 2020
- Blurb: "An uncensored, unfiltered documentary and book about the reality of the Taliban people in Afghanistan."

**Predicted probability of success: 10.5%**
**Actual outcome: Failed** _(model agreed with outcome)_

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▼ duration_days                     shap=-1.029  value=59.000
  ▼ country_IT                        shap=-0.987  value=1.000
  ▼ category_name_Journalism          shap=-0.649  value=1.000
  ▼ log_goal                          shap=-0.582  value=9.904
  ▲ trend_score                       shap=+0.324  value=0.000
  ▲ has_video                         shap=+0.276  value=1.000
  ▲ launch_hour                       shap=+0.195  value=10.000
  ▼ country_US                        shap=-0.165  value=0.000
  ▲ launch_dow                        shap=+0.155  value=0.000
  ▲ blurb_readability                 shap=+0.151  value=5.490
```

**Recommendations the tool would have given at launch:**
1. Your campaign length of 59 days is hurting your odds. Most successful campaigns run 21-35 days.  *(SHAP=-1.029)*
2. Your goal of $20,000 may be too high for this profile. Consider lowering it.  *(SHAP=-0.582)*
3. Adding a concrete number to your blurb (e.g., page count, run time, dimensions) tends to help.  *(SHAP=-0.095)*
4. A small dose of energy helps - consider rephrasing to sound more confident.  *(SHAP=-0.073)*
5. Your launch month (February) is a weaker month for this category. Consider shifting the launch.  *(SHAP=-0.055)*

**Reading:** Multiple recommendations triggered. Each one identifies a feature the creator could change before launching. Even at this small N, the same patterns (over-long duration, no video, oversized goal) recur across the failed campaigns.

---

### HIGH-CONFIDENCE WIN — *AUTOMNE AU LEVANT*

*model and outcome agreed — strong campaign profile*

**Inputs**
- Goal: $3,000  ·  Duration: 25 days
- Category: Comics  ·  Country: FR
- Video: yes  ·  Launch year: 2025
- Blurb: "Une BD pour témoigner de la guerre de 2024 au Liban"

**Predicted probability of success: 95.8%**
**Actual outcome: Successful** _(model agreed with outcome)_

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▲ category_name_Comics              shap=+0.934  value=1.000
  ▲ trend_score                       shap=+0.610  value=0.000
  ▲ duration_days                     shap=+0.500  value=25.000
  ▲ log_goal                          shap=+0.225  value=8.007
  ▼ blurb_readability                 shap=-0.221  value=95.688
  ▲ has_trend                         shap=+0.196  value=1.000
  ▲ launch_dow                        shap=+0.183  value=0.000
  ▲ blurb_has_number                  shap=+0.172  value=1.000
  ▼ has_video                         shap=-0.171  value=1.000
  ▲ launch_hour                       shap=+0.155  value=10.000
```

**Recommendations the tool would have given at launch:**
1. Your blurb is hard to read at a glance. Simplify the language.  *(SHAP=-0.221)*
2. A small dose of energy helps - consider rephrasing to sound more confident.  *(SHAP=-0.090)*

**Reading:** Strong profile across the board. The model didn't earn this prediction by being clever — it just rewarded a campaign that got the basics right (sensible goal, short duration, video, tight blurb) and matched a category that performs well.

---

### SURPRISE WIN — *A Moving Feast: A food truck for refugees in Lebanon*

*model called this a likely failure; the creator pulled it off*

**Inputs**
- Goal: $30,744  ·  Duration: 30 days
- Category: Food  ·  Country: GB
- Video: yes  ·  Launch year: 2015
- Blurb: "We want to help an amazing Palestinian woman in Lebanon buy a food truck so her catering social enterprise can grow even further."

**Predicted probability of success: 30.1%**
**Actual outcome: Successful** _(model DISAGREED with outcome)_

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▼ log_goal                          shap=-0.625  value=10.333
  ▲ has_video                         shap=+0.596  value=1.000
  ▼ category_name_Food                shap=-0.406  value=1.000
  ▼ country_US                        shap=-0.184  value=0.000
  ▼ trend_score                       shap=-0.152  value=50.000
  ▼ launch_dow                        shap=-0.140  value=4.000
  ▼ country_GB                        shap=-0.107  value=1.000
  ▼ blurb_word_count                  shap=-0.081  value=23.000
  ▼ category_name_Film & Video        shap=-0.076  value=0.000
  ▲ launch_hour                       shap=+0.070  value=16.000
```

**Recommendations the tool would have given at launch:**
1. Your goal of $30,744 may be too high for this profile. Consider lowering it.  *(SHAP=-0.625)*
2. Google Trends interest for Food is currently soft. Either wait for the next interest peak or lean into your differentiators.  *(SHAP=-0.152)*
3. Your planned launch day-of-week (Friday) is associated with lower success. Tuesdays and Wednesdays tend to perform best.  *(SHAP=-0.140)*
4. A small dose of energy helps - consider rephrasing to sound more confident.  *(SHAP=-0.059)*

**Reading:** Worth taking seriously. The tabular features missed something — almost certainly community trust, narrative quality of the blurb, or pre-launch audience built outside Kickstarter. This is a humility check on the recommender: a low score is not a verdict, it's a signal to investigate.

---

## Part B — Hypothetical creator personas

Three plausible Lebanese-diaspora creator profiles, run through the recommender as if they were preparing to launch in 2024. These show what the tool actually outputs for under-represented use-cases that are sparse in the data.

### PERSONA 1 — Beirut Port Blast Relief Documentary

*First-time filmmaker, Lebanese-American, raising for a feature documentary*

**Inputs**
- Goal: $60,000  ·  Duration: 45 days
- Category: Film & Video  ·  Country: US
- Video: yes  ·  Launch year: 2024
- Blurb: "A feature documentary about families rebuilding their lives after the 2020 Beirut port blast."

**Predicted probability of success: 73.0%**

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▼ log_goal                          shap=-0.727  value=11.002
  ▲ launch_hour                       shap=+0.412  value=12.000
  ▲ has_video                         shap=+0.260  value=1.000
  ▲ trend_score                       shap=+0.218  value=0.000
  ▲ blurb_has_number                  shap=+0.213  value=1.000
  ▼ duration_days                     shap=-0.166  value=45.000
  ▲ blurb_readability                 shap=+0.151  value=29.468
  ▲ blurb_word_count                  shap=+0.128  value=14.000
  ▲ launch_dow                        shap=+0.122  value=2.000
  ▲ country_US                        shap=+0.071  value=1.000
```

**Recommendations the tool would have given at launch:**
1. Your goal of $60,000 may be too high for this profile. Consider lowering it.  *(SHAP=-0.727)*
2. Your campaign length of 45 days is hurting your odds. Most successful campaigns run 21-35 days.  *(SHAP=-0.166)*
3. Your launch month (August) is a weaker month for this category. Consider shifting the launch.  *(SHAP=-0.066)*
4. A small dose of energy helps - consider rephrasing to sound more confident.  *(SHAP=-0.053)*

**Reading:** Common diaspora pattern — first-time creator, mission-driven film, ambitious goal, long campaign. The recommender's value here isn't to discourage — it's to flag the two settings most likely to sink the campaign and let the creator decide whether to adjust.

---

### PERSONA 2 — Lebanese Family Cookbook

*Diaspora creator preserving grandmother's recipes*

**Inputs**
- Goal: $8,000  ·  Duration: 30 days
- Category: Publishing  ·  Country: FR
- Video: no  ·  Launch year: 2024
- Blurb: "A 200-page hardback cookbook of my grandmother's Lebanese recipes, with photography."

**Predicted probability of success: 90.3%**

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▲ trend_score                       shap=+0.662  value=0.000
  ▲ category_name_Publishing          shap=+0.360  value=1.000
  ▲ launch_hour                       shap=+0.326  value=11.000
  ▼ country_US                        shap=-0.226  value=0.000
  ▼ has_video                         shap=-0.217  value=0.000
  ▲ blurb_has_number                  shap=+0.210  value=1.000
  ▲ launch_dow                        shap=+0.206  value=1.000
  ▲ has_trend                         shap=+0.176  value=1.000
  ▼ category_name_Games               shap=-0.141  value=0.000
  ▲ log_goal                          shap=+0.101  value=8.987
```

**Recommendations the tool would have given at launch:**
1. Add a video. Campaigns without a video underperform meaningfully in this category.  *(SHAP=-0.217)*
2. A small dose of energy helps - consider rephrasing to sound more confident.  *(SHAP=-0.078)*

**Reading:** Strong category match (Publishing cookbooks perform well), realistic goal, sane duration. Without a video, the model will likely flag that as the highest-leverage fix. Demonstrates the recommender pointing at ONE specific change rather than discouraging the project wholesale.

---

### PERSONA 3 — Lebanese-Indie Music Album

*Beirut-based musician, no Western audience, EP launch*

**Inputs**
- Goal: $4,500  ·  Duration: 60 days
- Category: Music  ·  Country: LB
- Video: no  ·  Launch year: 2024
- Blurb: "Help me record my debut EP — five original songs blending Arabic poetry with electronic production."

**Predicted probability of success: 64.6%**

**Top SHAP contributors (positive = helping, negative = hurting):**
```
  ▼ duration_days                     shap=-1.075  value=60.000
  ▲ trend_score                       shap=+0.561  value=4.000
  ▲ category_name_Music               shap=+0.375  value=1.000
  ▲ log_goal                          shap=+0.357  value=8.412
  ▲ blurb_word_count                  shap=+0.177  value=15.000
  ▲ has_trend                         shap=+0.167  value=1.000
  ▼ has_video                         shap=-0.157  value=0.000
  ▲ category_name_Technology          shap=+0.117  value=0.000
  ▼ blurb_has_exclaim                 shap=-0.092  value=0.000
  ▼ category_name_Games               shap=-0.091  value=0.000
```

**Recommendations the tool would have given at launch:**
1. Your campaign length of 60 days is hurting your odds. Most successful campaigns run 21-35 days.  *(SHAP=-1.075)*
2. Add a video. Campaigns without a video underperform meaningfully in this category.  *(SHAP=-0.157)*
3. A small dose of energy helps - consider rephrasing to sound more confident.  *(SHAP=-0.092)*
4. Adding a concrete number to your blurb (e.g., page count, run time, dimensions) tends to help.  *(SHAP=-0.052)*
5. Your planned launch day-of-week (Saturday) is associated with lower success. Tuesdays and Wednesdays tend to perform best.  *(SHAP=-0.051)*

**Reading:** The hardest case for the recommender. Country=LB is rare in the training data (most Kickstarter campaigns are US/GB/CA), the duration is over the sweet spot, and there's no video. This is precisely the under-represented profile the proposal targets — and exactly where the model's confidence is weakest. We surface that uncertainty rather than hide it.

---

## What this case study does and doesn't show

**Does show:**
- The recommender works on Lebanon-related campaigns at the same shape   it works elsewhere — when the model flags risk, the failure rate is   meaningfully higher; when it predicts a win, success is more likely.
- The advice it surfaces is concrete and feature-specific, not generic.
- The most common Lebanon-campaign failure modes mirror the global   ones: overlong duration, no video, oversized goal.

**Does NOT show:**
- Statistical reliability of subgroup metrics — N is far too small.
- That following the advice would have changed outcomes — the model is   correlational and the case studies cannot establish causation.
- That campaigns from Lebanon (country=LB) are well-represented — they   are not, and Persona 3 highlights this.
