#!/usr/bin/env python3
"""Generate synthetic CRM data for the GTM causal evaluation demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEGMENT_SHARES = {
    "sure_thing": 0.25,
    "persuadable": 0.20,
    "lost_cause": 0.45,
    "anti_persuadable": 0.10,
}

CONTROL_PROB = {
    "sure_thing": 0.18,
    "persuadable": 0.04,
    "lost_cause": 0.01,
    "anti_persuadable": 0.05,
}

TREATED_PROB = {
    "sure_thing": 0.19,
    "persuadable": 0.12,
    "lost_cause": 0.012,
    "anti_persuadable": 0.035,
}

SEGMENT_ORDER = list(SEGMENT_SHARES)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def segment_values(
    segments: np.ndarray,
    values: dict[str, float],
    noise: np.ndarray | None = None,
) -> np.ndarray:
    out = np.array([values[s] for s in segments], dtype=float)
    if noise is not None:
        out = out + noise
    return out


def generate_synthetic_crm(n_accounts: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    segments = rng.choice(
        SEGMENT_ORDER,
        size=n_accounts,
        p=[SEGMENT_SHARES[s] for s in SEGMENT_ORDER],
    )

    firmographic_base = {
        "sure_thing": 0.78,
        "persuadable": 0.60,
        "lost_cause": 0.30,
        "anti_persuadable": 0.48,
    }
    tech_base = {
        "sure_thing": 0.76,
        "persuadable": 0.68,
        "lost_cause": 0.34,
        "anti_persuadable": 0.50,
    }
    prior_activity_base = {
        "sure_thing": 1.00,
        "persuadable": 0.55,
        "lost_cause": 0.12,
        "anti_persuadable": 0.42,
    }

    firmographic_fit_score = clip01(
        segment_values(segments, firmographic_base, rng.normal(0, 0.12, n_accounts))
    )
    tech_fit_score = clip01(
        segment_values(segments, tech_base, rng.normal(0, 0.13, n_accounts))
    )
    prior_activity = np.clip(
        segment_values(segments, prior_activity_base, rng.normal(0, 0.22, n_accounts)),
        0,
        None,
    )

    log_company_size = rng.normal(
        5.2 + 1.0 * firmographic_fit_score + 0.25 * tech_fit_score,
        0.85,
        n_accounts,
    )
    company_size = np.clip(np.round(np.exp(log_company_size)), 20, 50_000).astype(int)
    annual_revenue_m = np.round(
        np.clip(company_size * rng.lognormal(mean=0.0, sigma=0.55, size=n_accounts) * 0.22, 1, None),
        2,
    )

    industry_probs = {
        "sure_thing": [0.25, 0.18, 0.18, 0.12, 0.15, 0.12],
        "persuadable": [0.20, 0.16, 0.22, 0.14, 0.16, 0.12],
        "lost_cause": [0.10, 0.14, 0.12, 0.25, 0.16, 0.23],
        "anti_persuadable": [0.14, 0.16, 0.12, 0.18, 0.18, 0.22],
    }
    industries = np.array(["software", "fintech", "marketplace", "healthcare", "retail", "other"])
    industry = np.array(
        [rng.choice(industries, p=industry_probs[s]) for s in segments],
        dtype=object,
    )
    region = rng.choice(
        np.array(["AMER", "EMEA", "APAC", "LATAM"], dtype=object),
        size=n_accounts,
        p=[0.55, 0.24, 0.16, 0.05],
    )

    prior_website_visits_90d = rng.poisson(0.8 + 8.0 * prior_activity + 3.0 * firmographic_fit_score)
    prior_email_engagement_90d = rng.poisson(0.2 + 2.6 * prior_activity + 0.8 * tech_fit_score)
    prior_sales_touches_180d = rng.poisson(0.5 + 3.2 * prior_activity + 1.2 * firmographic_fit_score)
    prior_opportunity_prob = sigmoid(-3.2 + 2.4 * firmographic_fit_score + 1.2 * prior_activity)
    prior_opportunity = rng.random(n_accounts) < prior_opportunity_prob

    stage_score = (
        0.9 * firmographic_fit_score
        + 0.55 * tech_fit_score
        + 0.28 * np.log1p(prior_website_visits_90d)
        + 0.45 * prior_opportunity
        + rng.normal(0, 0.35, n_accounts)
    )
    crm_stage_pre = np.select(
        [
            stage_score > 2.05,
            stage_score > 1.55,
            stage_score > 1.05,
        ],
        ["open_opportunity", "sales_accepted", "marketing_qualified"],
        default="unqualified",
    )

    linkedin_engagement_score = clip01(
        0.08
        + 0.42 * prior_activity
        + 0.22 * firmographic_fit_score
        + segment_values(
            segments,
            {"sure_thing": 0.09, "persuadable": 0.13, "lost_cause": 0.00, "anti_persuadable": 0.06},
        )
        + rng.normal(0, 0.14, n_accounts)
    )
    review_site_activity_score = clip01(
        0.05
        + 0.30 * tech_fit_score
        + 0.18 * prior_activity
        + segment_values(
            segments,
            {"sure_thing": 0.14, "persuadable": 0.19, "lost_cause": 0.02, "anti_persuadable": 0.08},
        )
        + rng.normal(0, 0.15, n_accounts)
    )
    intent_topic_count = rng.poisson(
        np.clip(0.2 + 2.6 * review_site_activity_score + 1.8 * linkedin_engagement_score, 0.05, None)
    )

    intent_latent = (
        -2.05
        + 2.3 * review_site_activity_score
        + 1.65 * linkedin_engagement_score
        + 0.65 * firmographic_fit_score
        + 0.18 * np.log1p(prior_website_visits_90d)
        + segment_values(
            segments,
            {"sure_thing": 0.72, "persuadable": 0.62, "lost_cause": 1.25, "anti_persuadable": 0.72},
        )
        + rng.normal(0, 0.70, n_accounts)
    )
    intent_score = np.round(100 * sigmoid(intent_latent), 1)
    high_intent = intent_score >= 80.0

    treatment_latent = (
        -2.65
        + 1.55 * high_intent.astype(float)
        + 1.15 * firmographic_fit_score
        + 0.16 * prior_sales_touches_180d
        + 0.55 * (crm_stage_pre == "marketing_qualified")
        + 0.80 * (crm_stage_pre == "sales_accepted")
        + 0.95 * (crm_stage_pre == "open_opportunity")
        + rng.normal(0, 0.35, n_accounts)
    )
    treatment_prob = sigmoid(treatment_latent)
    treated = rng.random(n_accounts) < treatment_prob

    delay_lambda = np.where(high_intent, 1.2, 3.4)
    outreach_delay_days = np.where(
        treated,
        np.clip(rng.poisson(delay_lambda, n_accounts), 0, 21),
        -1,
    )
    outreach_channel = np.full(n_accounts, "none", dtype=object)
    channel_draw = rng.choice(
        np.array(["email", "linkedin", "phone", "sequence"], dtype=object),
        size=n_accounts,
        p=[0.38, 0.24, 0.12, 0.26],
    )
    outreach_channel[treated] = channel_draw[treated]
    sdr_touches_14d = np.where(
        treated,
        rng.poisson(np.where(high_intent, 4.0, 2.1), n_accounts) + 1,
        0,
    )

    control_base = segment_values(segments, CONTROL_PROB)
    treated_base = segment_values(segments, TREATED_PROB)
    treatment_effect = treated_base - control_base
    feature_adjustment = (
        0.035 * (firmographic_fit_score - segment_values(segments, firmographic_base))
        + 0.018 * (tech_fit_score - segment_values(segments, tech_base))
        + 0.004 * np.log1p(prior_website_visits_90d)
        + 0.010 * prior_opportunity
        - 0.004 * np.maximum(prior_sales_touches_180d - 6, 0)
    )
    control_prob = np.clip(control_base + feature_adjustment, 0.002, 0.45)
    treated_prob = np.clip(control_prob + treatment_effect, 0.002, 0.45)
    outcome_prob = np.where(treated, treated_prob, control_prob)
    meeting_booked_60d = rng.random(n_accounts) < outcome_prob

    opportunity_prob = np.clip(
        0.12
        + 0.58 * meeting_booked_60d.astype(float)
        + 0.12 * firmographic_fit_score
        + 0.06 * prior_opportunity,
        0,
        0.92,
    )
    opportunity_created_90d = rng.random(n_accounts) < opportunity_prob
    pipeline_mean = np.exp(10.2 + 1.2 * firmographic_fit_score + rng.normal(0, 0.55, n_accounts))
    pipeline_created_90d = np.where(
        opportunity_created_90d,
        np.round(pipeline_mean / 1000) * 1000,
        0.0,
    )

    return pd.DataFrame(
        {
            "account_id": [f"acct_{i:06d}" for i in range(1, n_accounts + 1)],
            "campaign_id": "intent_window_2026q3",
            "segment_true": segments,
            "company_size": company_size,
            "annual_revenue_m": annual_revenue_m,
            "industry": industry,
            "region": region,
            "tech_fit_score": np.round(tech_fit_score, 4),
            "firmographic_fit_score": np.round(firmographic_fit_score, 4),
            "prior_website_visits_90d": prior_website_visits_90d,
            "prior_email_engagement_90d": prior_email_engagement_90d,
            "prior_sales_touches_180d": prior_sales_touches_180d,
            "prior_opportunity": prior_opportunity,
            "crm_stage_pre": crm_stage_pre,
            "intent_score": intent_score,
            "high_intent": high_intent,
            "intent_topic_count": intent_topic_count,
            "linkedin_engagement_score": np.round(linkedin_engagement_score, 4),
            "review_site_activity_score": np.round(review_site_activity_score, 4),
            "treated": treated,
            "outreach_delay_days": outreach_delay_days,
            "outreach_channel": outreach_channel,
            "sdr_touches_14d": sdr_touches_14d,
            "meeting_booked_60d": meeting_booked_60d,
            "opportunity_created_90d": opportunity_created_90d,
            "pipeline_created_90d": pipeline_created_90d,
            "control_meeting_prob_true": np.round(control_prob, 5),
            "treated_meeting_prob_true": np.round(treated_prob, 5),
            "treatment_effect_true": np.round(treated_prob - control_prob, 5),
        }
    )


def format_rate_table(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(group_cols, observed=True)[metric]
        .mean()
        .reset_index()
        .assign(**{metric: lambda x: x[metric].round(4)})
    )


def build_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"Rows: {len(df):,}")
    lines.append("")

    segment_share = (
        df["segment_true"]
        .value_counts(normalize=True)
        .reindex(SEGMENT_ORDER)
        .rename("share")
        .mul(100)
        .round(1)
        .reset_index()
        .rename(columns={"segment_true": "segment"})
    )
    lines.append("Segment shares (% of all accounts)")
    lines.append(segment_share.to_string(index=False))
    lines.append("")

    high_intent_mix = (
        df.loc[df["high_intent"], "segment_true"]
        .value_counts(normalize=True)
        .reindex(SEGMENT_ORDER)
        .rename("share")
        .mul(100)
        .round(1)
        .reset_index()
        .rename(columns={"segment_true": "segment"})
    )
    lines.append("High-intent composition (% of high-intent accounts)")
    lines.append(high_intent_mix.to_string(index=False))
    lines.append("")

    treatment_rates = format_rate_table(df, ["high_intent"], "treated")
    treatment_rates["treated"] = (treatment_rates["treated"] * 100).round(1)
    lines.append("Treatment rate by intent status (%)")
    lines.append(treatment_rates.to_string(index=False))
    lines.append("")

    conversion_by_intent = format_rate_table(df, ["high_intent"], "meeting_booked_60d")
    conversion_by_intent["meeting_booked_60d"] = (conversion_by_intent["meeting_booked_60d"] * 100).round(2)
    lines.append("Observed meeting rate by intent status (%)")
    lines.append(conversion_by_intent.to_string(index=False))
    lines.append("")

    conversion_by_treatment = format_rate_table(df, ["high_intent", "treated"], "meeting_booked_60d")
    conversion_by_treatment["meeting_booked_60d"] = (
        conversion_by_treatment["meeting_booked_60d"] * 100
    ).round(2)
    lines.append("Naive observed meeting rate by intent and treatment (%)")
    lines.append(conversion_by_treatment.to_string(index=False))
    lines.append("")

    true_te = (
        df.groupby("segment_true", observed=True)["treatment_effect_true"]
        .mean()
        .reindex(SEGMENT_ORDER)
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"segment_true": "segment", "treatment_effect_true": "true_lift_pp"})
    )
    lines.append("Planted true treatment effect by segment (percentage points)")
    lines.append(true_te.to_string(index=False))
    lines.append("")

    control_mix = (
        df.groupby("segment_true", observed=True)["control_meeting_prob_true"]
        .sum()
        .reindex(SEGMENT_ORDER)
    )
    control_mix = (control_mix / control_mix.sum() * 100).round(1)
    lines.append("Share of expected baseline/control meetings by segment (%)")
    lines.append(control_mix.reset_index().rename(columns={"segment_true": "segment", "control_meeting_prob_true": "share"}).to_string(index=False))

    return "\n".join(lines)


def write_outputs(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-accounts", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/generated/synthetic_crm.csv",
    )
    parser.add_argument("--summary", action="store_true", help="Print validation summary tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_synthetic_crm(n_accounts=args.n_accounts, seed=args.seed)
    write_outputs(df, args.output)
    if args.summary:
        print(build_summary(df))
        print("")
        print(f"Wrote {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
