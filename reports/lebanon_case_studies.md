# Lebanon-related campaigns — case-study appendix

This appendix runs all **49 campaigns** in the master table that matched the Lebanon-related keyword filter (`is_lebanon_related == 1`) through the pre-launch recommender. The sample is too small for reliable subgroup metrics, so we treat it qualitatively.

- Lebanon subgroup base rate: **65.3%** (vs full dataset 64.5%).

## Per-campaign predictions

- Model agreed with outcome (at threshold 0.5): **36/49**

### High-confidence misses (predicted ≥0.60, actually failed)

| name | category | country | goal | predicted | actual |
|---|---|---|---:|---:|---:|
| eka3@Beirut | Music | GB | $1,250 | 0.79 | 0 |
| Savore Shawarma Spices : The Pleasure Of Enjoying Flavors | Food | CA | $12,500 | 0.71 | 0 |
| Behind The Winds | Film & Video | US | $7,000 | 0.69 | 0 |
| Project Remembrance: An American Hero | Art | US | $20,000 | 0.66 | 0 |
| Norf: Web-Series | Film & Video | US | $3,000 | 0.64 | 0 |

### Correctly flagged risk (predicted <0.40, actually failed)

| name | category | country | goal | predicted | top recommendation |
|---|---|---|---:|---:|---|
| back to basics: authentic  lebanese  cuisine. | Food | AU | $50,000 | 0.02 | Your goal of $50,000 may be too high for this profile. Consider lowering it. |
| The Human Face of Terror | Journalism | IT | $20,000 | 0.11 | Your campaign length of 59 days is hurting your odds. Most successful campaigns run 21-35 days. |
| Ever wanted to send a friend a personalized onion? | Art | US | $3,000 | 0.14 | Your campaign length of 60 days is hurting your odds. Most successful campaigns run 21-35 days. |
| Imee's Lebanese Kitchen | Food | US | $25,000 | 0.21 | Your campaign length of 59 days is hurting your odds. Most successful campaigns run 21-35 days. |
| Log Cabins of Lebanon | Crafts | US | $20,000 | 0.29 | Your goal of $20,000 may be too high for this profile. Consider lowering it. |
| Anjar 1939-2019: Rebuilding Musa Dagh in Lebanon | Photography | IT | $5,000 | 0.30 | Add a video. Campaigns without a video underperform meaningfully in this category. |
| BEIRUT, LADY OF LEBANON | Theater | US | $30,000 | 0.34 | Your campaign length of 60 days is hurting your odds. Most successful campaigns run 21-35 days. |
| SAJisfaction: Food from the heart of Beirut | Food | CA | $6,500 | 0.35 | Add a video. Campaigns without a video underperform meaningfully in this category. |

### High-confidence wins (predicted ≥0.70, actually succeeded)

| name | category | country | goal | predicted |
|---|---|---|---:|---:|
| AUTOMNE AU LEVANT | Comics | FR | $3,000 | 0.96 |
| THE ROAD TO MARRAKESH ~ MEETING WITH OUR MOROCCAN SISTERS! | Music | US | $16,000 | 0.87 |
| ARSAL - A BORDERLAND TALE | Comics | FR | $13,000 | 0.87 |
| Help fund Flying Kebab #4 | Film & Video | US | $300 | 0.85 |
| ITISAL | Film & Video | GB | $6,000 | 0.84 |
| INSTRUMENT OF PEACE ~ LIBANA Returns to the Studio! | Music | US | $2,300 | 0.83 |
| We Are Not Princesses | Film & Video | US | $39,000 | 0.82 |
| Tabkh el Mama: A Lebanese Mother's Kitchen | Publishing | FR | $5,000 | 0.82 |
| No Woman's Land | Photography | CA | $35,000 | 0.81 |
| Peirene Now! No. 3: Shatila Stories | Publishing | GB | $8,500 | 0.80 |
