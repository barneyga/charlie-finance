# Crown Macro Letter — Finance Dashboard Insight Guide

Synthesized from 19 issues of The Crown Macro Letter by Nicholas Crown.
This document defines the analytical framework, key signals, data sources, and dashboard design priorities derived from the newsletter's recurring themes and methodology.

---

## 1. Crown's Core Analytical Philosophy

Understanding how Crown thinks is as important as knowing what he tracks. His framework rests on five principles your dashboard should reflect:

### 1.1 Price Tells the Truth, Headlines Lie
Crown consistently reads price action over narrative. A stock that stops going up on good news is more important than the good news itself. The dashboard should surface **narrative vs. price divergence**, not just price.

### 1.2 Relative Performance Beats Absolute
He almost never looks at assets in isolation. Everything is relative: gold vs. silver, equal-weight vs. cap-weight, US vs. international, defense vs. aerospace. The core dashboard question is always **"relative to what?"**

### 1.3 Breadth Reveals What the Index Hides
Cap-weighted indices can be at all-time highs while most stocks are declining. This "Sleight of Hand" is a recurring Crown theme. Breadth and equal-weight metrics must be first-class citizens in the dashboard.

### 1.4 Macro Regimes Drive Sector Outcomes
Crown identifies the prevailing macro regime (tariff regime, war regime, rate regime, dollar regime) and then asks which sectors benefit or suffer. The dashboard should make the current regime legible and show its sector implications.

### 1.5 Crowded Trades Are Risk, Not Confirmation
When everyone is in the same trade (NVDA, S&P 500, long USD), Crown treats it as a warning. Positioning and crowding signals should be visible alongside price.

---

## 2. Recurring Themes (Your Dashboard's Backbone)

These are the themes Crown returned to most across the 19 letters. Each should correspond to a dashboard section or view.

### Theme 1: Market Breadth & Index Illusion
**Letters:** "Sleight of Hand," "The Market's Only Bet," "The Most Crowded Trade"
- The S&P 500 (cap-weighted) regularly diverges from RSP (equal-weight). When SPY is up and RSP is flat or down, that's a concentration risk signal.
- The "market" is often just 7-10 stocks. The dashboard should always show how much of the index return is driven by the top holdings.
- **Key Insight Type:** Show SPY vs RSP performance spread, market cap concentration of top 10 holdings, % of S&P stocks above their 200-day MA.

### Theme 2: Cross-Asset Divergence (Stocks vs. Bonds)
**Letters:** "Growth Divide: Stocks Are Calm, Bonds Aren't," "When Good News Stops Working"
- When equities are calm and bonds are selling off (yields rising), the bond market is signaling stress the equity market hasn't priced yet. Crown treats this as a leading indicator of equity weakness.
- When positive earnings/macro data stops lifting stocks, that exhaustion signals trend reversal.
- **Key Insight Type:** Stock/bond correlation tracker, equity vs. bond yield divergence alerts, "good news reaction" tracker (how stocks respond to beats/misses).

### Theme 3: Metals — Gold, Silver, and Their Relationship
**Letters:** "Gold/Silver Disconnect," "Silver's Fakeout," "Why Metals Fell," "Black Gold"
- The gold/silver ratio is one of Crown's most-watched signals. A high ratio (gold outperforming silver) means risk-off/monetary demand. A falling ratio means silver is catching up, which can signal industrial demand returning or a speculative rotation.
- Silver moves are often misread as industrial demand when they're actually speculative or momentum. Crown distinguishes between the two by looking at industrial ETFs and base metals alongside silver.
- Gold declines are often driven by real rate moves (rising real yields = gold headwind) and dollar strength, not just sentiment.
- **Key Insight Type:** Live gold/silver ratio with historical bands, real rates vs. gold price overlay, silver vs. base metals (copper, zinc) divergence, gold vs. DXY relationship.

### Theme 4: Energy & Geopolitical Premium
**Letters:** "Black Gold: The War Is New, The Bid Isn't," "After the Shock: Modeling a Longer War"
- Oil has a persistent geopolitical bid that predates any specific conflict. New wars don't create demand from zero — they amplify an existing bid.
- Longer conflicts benefit energy (oil, LNG), defense, and metals (industrial + safe haven). They pressure bonds, consumer discretionary, and supply-chain-exposed equities.
- **Key Insight Type:** Oil price vs. geopolitical risk index overlay, energy sector (XLE) relative strength, war scenario impact matrix across asset classes.

### Theme 5: Sector Rotation & Scorecards
**Letters:** "The 2026 Sector Scorecard," "Factory Reset," "Most Defense Trades Are Actually Aerospace Trades"
- Crown tracks 11 GICS sectors and ranks them by YTD performance, relative momentum, and macro tailwind/headwind. He updates this scoring regularly.
- Within sectors, he goes sub-sector level: "defense" is really aerospace (ITA) + pure defense companies. QQQ is not "tech" — it's a mega-cap concentration vehicle that misses software (IGV) and semis (SOXX).
- **Key Insight Type:** Sector performance heatmap (weekly, monthly, YTD), sub-sector breakdown within Tech/Defense/Energy, rolling sector momentum scores, sector rotation wheel.

### Theme 6: AI & Tech Leadership Rotation
**Letters:** "Changing of the Guard," "NVDA's Illusion," "Why QQQ Is the Wrong Way to Own Tech"
- NVDA drove the AI trade but Crown identified the shift: META, GOOG, AMZN, and MSFT are taking AI leadership in returns while NVDA enthusiasm produces no stock upside (excitement without performance).
- QQQ is a poor proxy for tech exposure because of its mega-cap concentration. Better vehicles: IGV (software), SOXX (semis), individual names.
- **Key Insight Type:** NVDA vs. AI basket relative performance, QQQ vs. IGV vs. SOXX divergence tracker, AI capex spend vs. stock returns for hyperscalers.

### Theme 7: International vs. US Rotation
**Letters:** "The Shift: The Returns Weren't in the U.S.," "The Breakout No One Is Watching"
- Crown identified early that 2026 returns were migrating to Europe and international markets, driven by European defense spending, rearmament, and monetary divergence.
- European defense stocks (RHEINMETALL, Thales, BAE, Saab) were a specific breakout no mainstream analysis covered. The DAX broke out while US indices stalled.
- **Key Insight Type:** US vs. International (EFA, VGK, EEM) relative return tracker, country ETF performance table, European defense basket vs. US defense basket.

### Theme 8: Tariff & Fiscal Policy Impact
**Letters:** "Stimulus 2.0: Tariff Dividend," "Factory Reset"
- Crown theorized tariff revenue could be recycled as a "stimulus dividend" — effectively a fiscal transfer that could lift certain domestic sectors even as tariffs create inflation headwinds elsewhere.
- "Factory Reset" refers to the new trading pattern post-tariff regime: different sectors lead, correlations shift, old playbooks break.
- **Key Insight Type:** Tariff-sensitive sector tracker (domestic industrials, homebuilders vs. import-exposed retail/consumer), inflation breakeven monitor, fiscal impulse indicator.

### Theme 9: Crowding & Positioning Risk
**Letters:** "The Most Crowded Trade," "NVDA's Illusion," "The Market's Only Bet"
- When a single thesis drives the market (AI = NVDA = S&P going up), that concentration is itself the risk. The market becomes a one-variable system.
- **Key Insight Type:** COT (Commitment of Traders) positioning for key assets, ETF flow data, options skew as crowding proxy, breadth of participation in rallies.

---

## 3. Key Metrics & Data Points to Track

### Equities
| Metric | Purpose |
|---|---|
| SPY vs RSP spread (daily, weekly) | Breadth / concentration signal |
| % of S&P 500 stocks above 50/200-day MA | Market health |
| Top 10 holdings % of S&P weight | Concentration risk |
| Sector relative strength rankings (11 GICS) | Rotation detection |
| Equal-weight sector ETFs vs cap-weight | Sub-sector breadth |
| NVDA vs. AI basket (META, GOOG, MSFT, AMZN) | AI leadership |
| QQQ vs IGV vs SOXX | Tech sub-sector divergence |
| US (SPY) vs International (VGK, EFA, EEM) | Geographic rotation |
| European defense basket | Geopolitical trade |

### Fixed Income
| Metric | Purpose |
|---|---|
| 2Y, 10Y, 30Y Treasury yields | Rate regime |
| 2s10s yield curve | Recession signal |
| Real yield (10Y TIPS) | Gold/dollar driver |
| Equity vs. bond correlation (rolling 30/60d) | Risk regime |
| High yield spread (HYG vs IEI) | Credit stress |
| Inflation breakevens (5Y, 10Y) | Tariff/inflation regime |

### Commodities & Metals
| Metric | Purpose |
|---|---|
| Gold price (GLD) | Safe haven / monetary demand |
| Silver price (SLV) | Industrial vs. speculative |
| Gold/Silver ratio | Risk regime indicator |
| Copper price | Global growth proxy |
| Silver vs. Copper divergence | Silver signal quality |
| WTI / Brent crude | Energy / geopolitical |
| XLE relative to S&P | Energy sector momentum |
| Oil vs. defense sector correlation | War premium |

### Macro / Cross-Asset
| Metric | Purpose |
|---|---|
| DXY (US Dollar Index) | Dollar regime |
| Gold vs. DXY relationship | Metals context |
| Geopolitical risk index (GPR) | Conflict premium |
| COT data: Net positioning for SPX, gold, oil | Crowding detection |
| VIX vs. realized volatility | Complacency signal |
| Put/Call ratio | Sentiment |
| Global PMI (US, EU, China) | Growth regime |

---

## 4. Crown's Signal Types — Dashboard Alert Framework

Crown identifies several recurring signal types. Your dashboard should be able to surface these explicitly:

### Signal Type 1: "Good News Stops Working"
- **Definition:** Asset fails to rally on positive catalyst (earnings beat, good data)
- **Data needed:** Price reaction to catalyst events (earnings, CPI, jobs reports)
- **Dashboard alert:** Flag when SPX or sector ETF is down/flat on a day with strong positive macro prints

### Signal Type 2: Cross-Asset Divergence
- **Definition:** Two historically correlated assets moving in opposite directions
- **Examples:** Stocks up, bonds selling off. Gold up, silver flat. SPY up, RSP flat.
- **Dashboard alert:** Rolling correlation monitor with divergence alerts when spread exceeds N standard deviations

### Signal Type 3: Regime Change / "Factory Reset"
- **Definition:** Old leadership stops working, new sectors take over
- **Data needed:** 90-day rolling sector return rankings, look for rank reversals
- **Dashboard alert:** When a top-3 sector drops to bottom-3 or vice versa over rolling window

### Signal Type 4: Breakout in Ignored Asset
- **Definition:** An asset outside consensus attention makes a technical breakout
- **Examples:** European defense stocks, the DAX, small-cap defense names
- **Dashboard alert:** 52-week high scanner filtered by assets with low media/search attention (Google Trends proxy)

### Signal Type 5: Crowded Trade Unwinding
- **Definition:** A universally-owned trade starts to crack
- **Data needed:** COT positioning, ETF flows, price action relative to positioning
- **Dashboard alert:** When net long positioning is at 90th+ percentile and price breaks below 20-day MA

### Signal Type 6: Narrative vs. Price Divergence
- **Definition:** The story says one thing, the chart says another
- **Examples:** NVDA excitement but stock flat/down. "Defense boom" but defense ETFs underperforming
- **Dashboard alert:** Sentiment score (news volume/tone) vs. 30-day price return divergence

---

## 5. Dashboard Architecture Recommendation

Based on Crown's framework, here's a suggested view structure:

### View 1: Macro Regime Dashboard (Homepage)
- Current regime identification: Rate regime, dollar regime, geopolitical regime, growth regime
- Key cross-asset readings: SPY, TLT, GLD, DXY, OIL, VIX
- Top divergences of the week
- Regime change probability score

### View 2: Breadth & Concentration Monitor
- SPY vs RSP spread (chart + current reading)
- % stocks above 50/200-day MA
- S&P concentration: top 5, top 10 weight
- Advance/decline line
- New highs vs. new lows

### View 3: Sector Scorecard
- 11 GICS sectors ranked by: 1-week, 1-month, 3-month, YTD return
- Color-coded heatmap
- Sub-sector breakdown for Tech (IGV/SOXX/QQQ), Defense (ITA/pure defense), Energy
- Sector relative strength momentum score

### View 4: Metals & Commodities
- Gold, silver, copper, WTI prices
- Gold/silver ratio with historical percentile
- Real yields vs gold overlay
- Silver vs copper divergence
- Energy sector (XLE) relative strength

### View 5: Cross-Asset Stress Monitor
- Stock/bond correlation (rolling 30d)
- Yield curve (2s10s)
- Credit spread (HY)
- VIX vs. realized vol
- Equity/bond divergence alert

### View 6: Geographic Rotation
- US vs International return comparison (SPY, VGK, EFA, EEM, DAX)
- Country ETF performance table
- European defense basket tracker
- DXY and its impact on international returns

### View 7: AI & Tech Leadership
- NVDA vs AI basket (META, GOOG, MSFT, AMZN) relative performance
- QQQ vs IGV vs SOXX
- AI capex announcements vs. stock reaction

### View 8: Crowding & Positioning
- COT net positioning for S&P, gold, oil, EUR/USD
- ETF flow tracker (weekly inflows/outflows by sector)
- Options put/call ratios
- Google Trends for key financial terms (proxy for retail crowding)

---

## 6. Crown's Vocabulary — Terms to Use in UI

To make the dashboard feel aligned with the analytical voice of the newsletter, use Crown's language in labels, tooltips, and insight cards:

| Crown's Term | What It Means |
|---|---|
| "The bid" | Persistent demand/buying in an asset |
| "Changing of the guard" | Leadership rotation between assets/sectors |
| "Sleight of hand" | Index disguising weakness underneath |
| "Factory reset" | Regime change resetting which trades work |
| "Crowded trade" | Consensus position with reversal risk |
| "Good news stops working" | Bullish exhaustion signal |
| "The setup" | A compelling risk/reward configuration |
| "Breakout" | Price clearing a key technical level |
| "Fakeout" | False breakout that reverses |
| "The dividend" | In tariff context, the fiscal stimulus effect of tariff revenue |
| "War premium" | Geopolitical risk embedded in asset prices |

---

## 7. Insight Card Templates

Crown structures each letter the same way. Use this structure for automated insight cards in the dashboard:

```
[SIGNAL TYPE] in [ASSET/SECTOR]

What's happening: [One sentence price/data observation]
Why it matters: [The macro/structural reason]
What to watch: [The indicator or catalyst that confirms/denies]
Historical analogue: [When has this happened before]
Risk if wrong: [What would invalidate the thesis]
```

Example:
```
DIVERGENCE in Metals

What's happening: Gold is up 8% YTD while silver is flat — ratio at 89, near decade highs.
Why it matters: High gold/silver ratio signals monetary/safe-haven demand dominating over
industrial. When industrial demand returns, silver typically catches up fast.
What to watch: Silver breaking above $32 on strong copper/PMI backdrop = catch-up trade.
Historical analogue: 2020 — ratio peaked at 120, then silver rallied 140% in 6 months.
Risk if wrong: Continued dollar strength and weak China PMI would keep silver lagging.
```

---

## 8. Prioritization for MVP

If you're building iteratively, here's the order Crown's framework suggests by impact:

1. **Breadth monitor** (SPY vs RSP, % above MAs) — most recurring theme, easiest to miss
2. **Sector scorecard** (11 sectors, weekly updated) — he references this constantly
3. **Metals dashboard** (gold/silver ratio, real yields) — deeply covered, quantitative
4. **Cross-asset divergence alerts** (stocks vs bonds) — his leading indicator framework
5. **Geographic rotation tracker** (US vs international) — his biggest 2026 call
6. **AI/Tech sub-sector breakdown** (NVDA vs basket, QQQ vs IGV/SOXX) — high reader interest
7. **Crowding/positioning monitor** (COT, ETF flows) — sophisticated but high value
8. **Energy/geopolitical premium** (oil, XLE, geopolitical risk) — war-driven, timely

---

*Document generated March 16, 2026. Based on 19 issues of The Crown Macro Letter.*
*Update this document as new letters are added to crown-letters/.*
