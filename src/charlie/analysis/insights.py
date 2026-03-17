"""Auto-generated Crown Macro insight cards.

Scans multiple signal types and produces human-readable cards in the style
of Nicholas Crown's macro newsletter. Each scanner returns an InsightCard
or None if no signal is active.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from charlie.storage.db import Database

logger = logging.getLogger(__name__)


@dataclass
class InsightCard:
    signal_type: str      # DIVERGENCE, BREADTH, CROWDED_TRADE, REGIME_SHIFT, EXHAUSTION, COMPLACENCY
    asset: str            # "Gold/Silver", "SPY", "Energy", etc.
    crown_term: str       # "Sleight of Hand", "Crowded Trade", "Factory Reset", etc.
    headline: str         # One-liner
    whats_happening: str
    why_it_matters: str
    what_to_watch: str
    risk_if_wrong: str
    severity: float       # 0-1 for ranking
    section_link: str     # section_id to jump to


# ── Scanner functions ─────────────────────────────────────────────


def _scan_divergence(db: Database) -> InsightCard | None:
    """Detect cross-asset divergences: gold/silver, SPY-TLT, oil/gold."""
    try:
        from charlie.analysis.derived import gold_silver_ratio, stock_bond_correlation, oil_gold_ratio
        from charlie.analysis.stats import rolling_zscore

        # Gold/Silver ratio z-score
        gsr = gold_silver_ratio(db)
        if not gsr.empty and len(gsr) >= 252:
            gsr_z = rolling_zscore(gsr, 252)
            if not gsr_z.empty:
                z = float(gsr_z.iloc[-1])
                val = float(gsr.iloc[-1])
                if abs(z) > 1.5:
                    direction = "surging" if z > 0 else "collapsing"
                    return InsightCard(
                        signal_type="DIVERGENCE",
                        asset="Gold/Silver",
                        crown_term="Safe Haven Signal",
                        headline=f"Gold/Silver ratio {direction} (z={z:.1f}) at {val:.1f}",
                        whats_happening=f"The gold/silver ratio is at {val:.1f}, which is {abs(z):.1f} standard deviations {'above' if z > 0 else 'below'} its 1-year average. Gold is {'massively outperforming' if z > 0 else 'underperforming'} silver.",
                        why_it_matters="A rising gold/silver ratio signals flight to safety — investors prefer the 'harder' monetary metal. Falling ratio means risk appetite is returning.",
                        what_to_watch="Watch for the ratio to mean-revert. Extremes above 2 z-scores historically reverse within weeks.",
                        risk_if_wrong="If geopolitical risk escalates further, the ratio can stay elevated longer than expected.",
                        severity=min(abs(z) / 3.0, 1.0),
                        section_link="metals",
                    )

        # SPY-TLT correlation turning positive
        corr = stock_bond_correlation(db)
        if not corr.empty:
            c = float(corr.iloc[-1])
            if c > 0.3:
                return InsightCard(
                    signal_type="DIVERGENCE",
                    asset="SPY/TLT",
                    crown_term="Correlation Regime Break",
                    headline=f"Stocks and bonds moving together (corr={c:.2f})",
                    whats_happening=f"The 63-day rolling correlation between SPY and TLT is {c:.2f}. Normally negative (diversification works), positive correlation means both are selling off together.",
                    why_it_matters="When 60/40 stops working, there is nowhere to hide. This usually happens during inflation scares or Fed tightening cycles.",
                    what_to_watch="Monitor if correlation stays above 0.2. A return to negative territory restores normal portfolio diversification.",
                    risk_if_wrong="Correlation can spike temporarily during liquidation events and quickly normalize.",
                    severity=min(c / 0.6, 1.0),
                    section_link="divergence",
                )
    except Exception as e:
        logger.debug("Divergence scan error: %s", e)
    return None


def _scan_breadth(db: Database) -> InsightCard | None:
    """Detect narrow breadth / 'Sleight of Hand'."""
    try:
        from charlie.analysis.derived import spy_rsp_spread, breadth_above_200d_ma
        from charlie.analysis.stats import rolling_zscore

        # SPY/RSP spread z-score
        ratio = spy_rsp_spread(db)
        if not ratio.empty and len(ratio) >= 252:
            z = rolling_zscore(ratio, 252)
            if not z.empty:
                z_val = float(z.iloc[-1])
                if z_val > 1.5:
                    breadth = breadth_above_200d_ma(db)
                    pct = breadth.get("current_pct", 50) if breadth.get("available") else 50
                    return InsightCard(
                        signal_type="BREADTH",
                        asset="SPY vs RSP",
                        crown_term="Sleight of Hand",
                        headline=f"SPY/RSP concentration z-score at {z_val:.1f} — only {pct:.0f}% of ETFs above 200d MA",
                        whats_happening=f"The cap-weighted S&P 500 is dramatically outperforming equal-weight (z={z_val:.1f}). Only {pct:.0f}% of tracked ETFs are above their 200-day moving average.",
                        why_it_matters="A handful of mega-caps are masking broad weakness. Crown calls this 'Sleight of Hand' — the index looks fine but the average stock is struggling.",
                        what_to_watch="Watch for RSP to start outperforming SPY, which would signal breadth broadening. Also watch small-cap IWM relative to SPY.",
                        risk_if_wrong="Mega-cap concentration can persist longer than expected in AI-driven narratives. It doesn't mean the market crashes — just that it's fragile.",
                        severity=min(z_val / 3.0, 1.0),
                        section_link="breadth",
                    )

        # Breadth below 50% even without SPY/RSP divergence
        breadth = breadth_above_200d_ma(db)
        if breadth.get("available") and breadth["current_pct"] < 40:
            pct = breadth["current_pct"]
            return InsightCard(
                signal_type="BREADTH",
                asset="Market Breadth",
                crown_term="Breadth Warning",
                headline=f"Only {pct:.0f}% of ETFs above 200d MA — critical breadth weakness",
                whats_happening=f"Just {breadth['above_count']} of {breadth['total_count']} tracked ETFs are trading above their 200-day moving average ({pct:.0f}%).",
                why_it_matters="When fewer than 40% of sectors are in uptrends, the market's structural support is weak. Rallies on thin breadth tend to fail.",
                what_to_watch="Watch for breadth to recover above 50%. Key signal: cyclical sectors (XLI, XLF, XLY) crossing back above their 200d MAs.",
                risk_if_wrong="Breadth can stay weak during sector rotations where leadership simply shifts rather than broadens.",
                severity=max(0.5, 1.0 - pct / 100),
                section_link="breadth",
            )
    except Exception as e:
        logger.debug("Breadth scan error: %s", e)
    return None


def _scan_crowded_trade(db: Database) -> InsightCard | None:
    """Detect crowded trade unwinding."""
    try:
        from charlie.analysis.derived import crowded_trade_unwind

        results = crowded_trade_unwind(db)
        unwinding = [r for r in results if r["unwinding"]]

        if unwinding:
            names = ", ".join(r["contract"] for r in unwinding)
            worst = max(unwinding, key=lambda r: abs(r["z_score"]))
            signal = "long unwinding" if worst["signal_type"] == "crowded_long_unwinding" else "short squeeze"

            return InsightCard(
                signal_type="CROWDED_TRADE",
                asset=names,
                crown_term="The Most Crowded Trade",
                headline=f"Crowded {signal} detected: {names}",
                whats_happening=f"{worst['contract']} has a positioning z-score of {worst['z_score']:+.1f} while price is {abs(worst['price_vs_ma_pct']):.1f}% {'below' if worst['price_vs_ma_pct'] < 0 else 'above'} its 20d MA.",
                why_it_matters="When extreme positioning meets adverse price action, forced liquidation accelerates the move. Crown's principle: crowded trades are risk, not confirmation.",
                what_to_watch="Watch for positioning to normalize (z-score returning toward 0). The unwind is over when pain has forced enough capitulation.",
                risk_if_wrong="Not every extreme position unwinds violently. Strong fundamental drivers can sustain extreme positioning longer.",
                severity=min(abs(worst["z_score"]) / 3.0, 1.0),
                section_link="cot",
            )
    except Exception as e:
        logger.debug("Crowded trade scan error: %s", e)
    return None


def _scan_regime_shift(db: Database) -> InsightCard | None:
    """Detect sector rank reversals — 'Factory Reset'."""
    try:
        from charlie.analysis.derived import sector_rank_reversal

        result = sector_rank_reversal(db)
        if not result.get("available") or not result["has_reversal"]:
            return None

        reversals = result["reversals"]
        descriptions = []
        for r in reversals:
            direction = "surged" if r["rank_change"] > 0 else "collapsed"
            descriptions.append(f"{r['name']} {direction} #{r['previous_rank']} -> #{r['current_rank']}")

        headline_parts = descriptions[:2]
        headline = "; ".join(headline_parts)

        return InsightCard(
            signal_type="REGIME_SHIFT",
            asset="Sectors",
            crown_term="Factory Reset",
            headline=f"Factory Reset: {headline}",
            whats_happening=f"{len(reversals)} sector(s) had major rank reversals over the past quarter: {'; '.join(descriptions)}.",
            why_it_matters="Major sector rank changes signal institutional rotation — the market is repricing which parts of the economy will lead. Crown calls this a 'Factory Reset' when old leaders become laggards.",
            what_to_watch="Watch if the new leaders sustain their gains over the next month. Sustained rotation confirms regime change; a snap-back means it was noise.",
            risk_if_wrong="Short-term rank changes can be driven by one-off events (earnings, policy) rather than structural shifts.",
            severity=min(len(reversals) * 0.3 + 0.4, 1.0),
            section_link="sectors",
        )
    except Exception as e:
        logger.debug("Regime shift scan error: %s", e)
    return None


def _scan_exhaustion(db: Database, api_key: str, releases) -> InsightCard | None:
    """Detect 'Good News Stops Working' — market exhaustion."""
    try:
        from charlie.analysis.derived import exhaustion_signal

        result = exhaustion_signal(db, api_key, releases)
        if not result.get("available"):
            return None

        if result["signal"] not in ("caution", "exhaustion"):
            return None

        score = result["score"]
        neg = result["count_negative"]
        total = result["count_total"]

        return InsightCard(
            signal_type="EXHAUSTION",
            asset="SPY",
            crown_term="Good News Stops Working",
            headline=f"Market sold off on {neg}/{total} recent data releases (score: {score:.0%})",
            whats_happening=f"Of the last {total} high-impact economic releases, {neg} saw negative SPY returns on the release day. This is {'well above' if score > 0.6 else 'above'} the normal rate.",
            why_it_matters="When markets sell off on objectively positive data, the move is exhausted. Expectations have run too far ahead of fundamentals. Crown's key signal: good news that doesn't lift prices is the most bearish signal.",
            what_to_watch="Watch for a positive surprise that actually moves markets higher — that would reset the exhaustion signal. Until then, rallies are selling opportunities.",
            risk_if_wrong="Sometimes markets sell on good news because they're pricing in policy tightening (good economy = higher rates), not exhaustion.",
            severity=min(score, 1.0),
            section_link="regime",
        )
    except Exception as e:
        logger.debug("Exhaustion scan error: %s", e)
    return None


def _scan_complacency(db: Database) -> InsightCard | None:
    """Detect VIX vs realized vol complacency."""
    try:
        from charlie.analysis.derived import vix_vs_realized_vol

        result = vix_vs_realized_vol(db)
        if not result.get("available"):
            return None

        premium = result["premium"]
        signal = result["signal"]

        if signal == "complacency":
            return InsightCard(
                signal_type="COMPLACENCY",
                asset="VIX",
                crown_term="Vol Complacency",
                headline=f"VIX premium negative ({premium:+.1f}) — implied vol below realized",
                whats_happening=f"VIX is at {result['vix']:.1f} while realized volatility is {result['realized_vol']:.1f}. The VIX premium is {premium:+.1f}, meaning options are priced below actual recent volatility.",
                why_it_matters="When implied vol drops below realized vol, the market is complacent — not pricing in risk. This is historically followed by volatility spikes. Cheap protection is an opportunity.",
                what_to_watch="Watch for VIX to snap back above realized vol. The premium returning to positive territory would signal the complacency phase is ending.",
                risk_if_wrong="VIX can stay suppressed in strong bull markets with low macro uncertainty. Complacency alone isn't a timing signal.",
                severity=min(abs(premium) / 5.0 + 0.3, 1.0),
                section_link="credit",
            )

        if signal == "fear_overshoot":
            return InsightCard(
                signal_type="COMPLACENCY",
                asset="VIX",
                crown_term="Fear Overshoot",
                headline=f"VIX premium extreme ({premium:+.1f}) — panic pricing",
                whats_happening=f"VIX is at {result['vix']:.1f} while realized volatility is {result['realized_vol']:.1f}. The VIX premium is {premium:+.1f}, meaning options are priced far above actual recent volatility.",
                why_it_matters="When implied vol massively exceeds realized vol, the market is pricing in a fear scenario. These extremes often mark near-term bottoms for risk assets.",
                what_to_watch="Watch for the VIX premium to start contracting. Selling put protection at these levels has historically been profitable.",
                risk_if_wrong="If the feared scenario materializes, realized vol catches up to implied vol and the premium normalizes from the wrong direction.",
                severity=min(premium / 15.0 + 0.3, 1.0),
                section_link="credit",
            )
    except Exception as e:
        logger.debug("Complacency scan error: %s", e)
    return None


# ── Entry point ───────────────────────────────────────────────────


def generate_insights(
    db: Database,
    api_key: str = "",
    releases: list | tuple = (),
) -> list[InsightCard]:
    """Run all scanners and return top 5 insights sorted by severity."""
    cards: list[InsightCard] = []

    # Run each scanner, collect non-None results
    scanners = [
        lambda: _scan_divergence(db),
        lambda: _scan_breadth(db),
        lambda: _scan_crowded_trade(db),
        lambda: _scan_regime_shift(db),
        lambda: _scan_exhaustion(db, api_key, releases),
        lambda: _scan_complacency(db),
    ]

    for scanner in scanners:
        try:
            card = scanner()
            if card is not None:
                cards.append(card)
        except Exception as e:
            logger.debug("Scanner failed: %s", e)

    # Sort by severity descending, return top 5
    cards.sort(key=lambda c: c.severity, reverse=True)
    return cards[:5]
