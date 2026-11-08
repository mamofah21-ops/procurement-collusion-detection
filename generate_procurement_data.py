"""
generate_procurement_data.py

Synthetic relational public procurement dataset with planted bid-rigging cartels,
designed to be mapped into a heterogeneous graph (company, contract, owner nodes).

Outputs (in ./data):
    companies.csv                 500 companies
    bids.csv                      5,000 bids
    ground_truth_companies.csv    cartel membership per company (EVALUATION ONLY)
    ground_truth_contracts.csv    which contracts were rigged (EVALUATION ONLY)

Planted collusion pattern (per cartel):
    * all members share one ultimate_beneficial_owner_id
    * members were registered within a short window of each other
    * every member bids on every cartel contract
    * bid amounts are near identical (cover bids 0.2% to 1.5% above the winner)
    * bids are submitted within minutes of each other, close to the deadline
    * the win rotates among members (bid rotation)

Realism / noise so the problem is not trivial:
    * risk_flag marks only part of each cartel, plus a few innocent firms (label noise)
    * legitimate owners hold 2 to 5 firms that never collude (shared UBO != fraud)
    * group sizes overlap: cartels range from 2 to 6 members, so the number of
      firms under one owner cannot separate cartels from honest groups
    * half of the honest groups stay in one sector and half registered their firms
      within months of each other, so sector and registration timing are not
      clean separators either
    * cartel members also bid normally on ordinary contracts

v2 (Nov 2026): v1 used cartels of 3 to 6 and honest owners with exactly two firms,
which let owner group size separate the classes perfectly. Fixed by overlapping sizes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_COMPANIES = 500
N_BIDS = 5000

CARTEL_SIZES = [2, 3, 3, 4, 5, 6]   # members per cartel (one entry per cartel)
N_CARTELS = len(CARTEL_SIZES)
CONTRACTS_PER_CARTEL = (10, 18)     # rigged contracts per cartel (inclusive)
LABEL_RATE_IN_CARTELS = 0.5         # share of cartel members already known (risk_flag = 1)
N_FLAGGED_OUTSIDERS = 3             # innocent firms wrongly flagged (label noise)
LEGIT_GROUP_SIZES = [2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6]   # firms per honest multi-firm owner

TENDER_START = pd.Timestamp("2019-01-01")
TENDER_END = pd.Timestamp("2024-12-31")
REG_START = pd.Timestamp("2000-01-01")
REG_END = pd.Timestamp("2018-12-31")

OUT_DIR = "data"

SECTORS = ["Construction", "Health Supplies", "ICT", "Education",
           "Energy", "Agriculture", "Transport", "Consultancy"]
SECTOR_WEIGHTS = [0.22, 0.14, 0.14, 0.12, 0.10, 0.10, 0.10, 0.08]

# log-normal parameters for contract value by sector (mean of log, sd of log)
SECTOR_VALUE_PARAMS = {
    "Construction": (13.2, 0.9), "Health Supplies": (12.0, 0.8),
    "ICT": (12.2, 0.9), "Education": (11.6, 0.7), "Energy": (13.0, 1.0),
    "Agriculture": (11.4, 0.7), "Transport": (12.4, 0.9), "Consultancy": (11.2, 0.6),
}

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_dates(start: pd.Timestamp, end: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    span = (end - start).days
    return start + pd.to_timedelta(rng.integers(0, span, n), unit="D")


def make_company_names(n: int) -> list[str]:
    prefixes = ["Ak", "Ben", "Kwa", "Ade", "Nana", "Osei", "Asa", "Ama", "Kofi", "Yaw",
                "Esi", "Abena", "Danso", "Owu", "Bona", "Seke", "Tano", "Volta", "Ashan", "Bri"]
    cores = ["tech", "build", "link", "star", "prime", "gold", "royal", "unity",
             "crest", "bridge", "field", "stone", "wave", "peak", "core"]
    suffixes = ["Ltd", "Enterprise", "Ventures", "Company Ltd", "Holdings",
                "Services", "Group", "Consult"]
    names: set[str] = set()
    while len(names) < n:
        names.add(f"{rng.choice(prefixes)}{rng.choice(cores)} {rng.choice(suffixes)}")
    out = list(names)
    rng.shuffle(out)
    return out


def contract_value(sector: str) -> float:
    mu, sd = SECTOR_VALUE_PARAMS[sector]
    return float(np.round(rng.lognormal(mu, sd), 2))


def tender_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    pub = random_dates(TENDER_START, TENDER_END - pd.Timedelta(days=60), 1)[0]
    pub = pub + pd.Timedelta(hours=int(rng.integers(8, 17)))
    deadline = pub + pd.Timedelta(days=int(rng.integers(14, 46)))
    return pub, deadline


# ---------------------------------------------------------------------------
# 1. Companies
# ---------------------------------------------------------------------------
def build_companies() -> pd.DataFrame:
    company_ids = [f"CMP{i:04d}" for i in range(1, N_COMPANIES + 1)]
    companies = pd.DataFrame({
        "company_id": company_ids,
        "company_name": make_company_names(N_COMPANIES),
        "registration_date": random_dates(REG_START, REG_END, N_COMPANIES),
        "ultimate_beneficial_owner_id": [f"UBO{i:04d}" for i in range(1, N_COMPANIES + 1)],
        "sector": rng.choice(SECTORS, size=N_COMPANIES, p=SECTOR_WEIGHTS),
        "risk_flag": 0,
        "cartel_id": pd.NA,   # dropped before export, kept for ground truth
    })
    return companies


def plant_cartels(companies: pd.DataFrame) -> dict[str, list[str]]:
    """Assign cartel members, shared UBOs and clustered registration dates."""
    cartels: dict[str, list[str]] = {}
    used: set[str] = set()
    # cartels lean toward high-value sectors
    cartel_sectors = rng.choice(
        ["Construction", "Health Supplies", "Energy", "ICT", "Transport"],
        size=N_CARTELS, p=[0.35, 0.2, 0.2, 0.15, 0.1],
    )

    for k, sector in enumerate(cartel_sectors, start=1):
        cartel_id = f"CARTEL_{k:02d}"
        size = CARTEL_SIZES[k - 1]
        pool = companies.loc[(companies["sector"] == sector)
                             & (~companies["company_id"].isin(used)), "company_id"].to_numpy()
        members = list(rng.choice(pool, size=size, replace=False))
        used.update(members)
        cartels[cartel_id] = members

        idx = companies["company_id"].isin(members)
        shared_ubo = companies.loc[idx, "ultimate_beneficial_owner_id"].iloc[0]
        companies.loc[idx, "ultimate_beneficial_owner_id"] = shared_ubo
        companies.loc[idx, "cartel_id"] = cartel_id

        # shell-company signal: registered within ~90 days of each other
        base = random_dates(pd.Timestamp("2014-01-01"), pd.Timestamp("2018-06-30"), 1)[0]
        companies.loc[idx, "registration_date"] = base + pd.to_timedelta(
            rng.integers(0, 90, size), unit="D")

        # only part of each cartel is already known to investigators
        known = rng.random(size) < LABEL_RATE_IN_CARTELS
        companies.loc[companies["company_id"].isin(np.array(members)[known]), "risk_flag"] = 1

    return cartels


def add_noise(companies: pd.DataFrame) -> None:
    """Honest multi-firm owners (overlapping cartel sizes) and a few wrongly flagged firms."""
    available = set(companies.loc[companies["cartel_id"].isna(), "company_id"])

    for size in LEGIT_GROUP_SIZES:
        pool = np.array(sorted(available))
        if rng.random() < 0.5:   # half of honest groups operate within one sector
            sector = rng.choice(SECTORS, p=SECTOR_WEIGHTS)
            in_sector = companies.loc[companies["company_id"].isin(available)
                                      & (companies["sector"] == sector), "company_id"].to_numpy()
            if len(in_sector) >= size:
                pool = np.sort(in_sector)
        group = rng.choice(pool, size=size, replace=False)
        available -= set(group)

        idx = companies["company_id"].isin(group)
        owner = companies.loc[idx, "ultimate_beneficial_owner_id"].iloc[0]
        companies.loc[idx, "ultimate_beneficial_owner_id"] = owner

        if rng.random() < 0.5:   # half registered their firms close together
            base = random_dates(pd.Timestamp("2005-01-01"), pd.Timestamp("2017-12-31"), 1)[0]
            companies.loc[idx, "registration_date"] = base + pd.to_timedelta(
                rng.integers(0, 180, size), unit="D")

    false_flags = rng.choice(np.array(sorted(available)), size=N_FLAGGED_OUTSIDERS, replace=False)
    companies.loc[companies["company_id"].isin(false_flags), "risk_flag"] = 1


# ---------------------------------------------------------------------------
# 2. Bids
# ---------------------------------------------------------------------------
def cartel_bids(companies: pd.DataFrame, cartels: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    bids, contracts = [], []
    sector_of = companies.set_index("company_id")["sector"]

    for cartel_id, members in cartels.items():
        sector = sector_of[members[0]]
        outsiders_pool = companies.loc[(companies["sector"] == sector)
                                       & (~companies["company_id"].isin(members)),
                                       "company_id"].to_numpy()
        n_contracts = int(rng.integers(CONTRACTS_PER_CARTEL[0], CONTRACTS_PER_CARTEL[1] + 1))
        rotation = list(rng.permutation(members))

        for j in range(n_contracts):
            key = f"{cartel_id}_{j}"
            value = contract_value(sector)
            pub, deadline = tender_window()
            contracts.append({"contract_key": key, "publication_date": pub,
                              "sector": sector, "is_rigged": 1, "cartel_id": cartel_id})

            winner = rotation[j % len(rotation)]           # bid rotation
            win_amount = value * rng.uniform(0.97, 1.02)   # priced near or above estimate
            burst = deadline - pd.Timedelta(minutes=int(rng.integers(60, 48 * 60)))

            for m in members:
                amount = win_amount if m == winner else win_amount * (1 + rng.uniform(0.002, 0.015))
                bids.append({
                    "contract_key": key, "company_id": m, "bid_amount": round(amount, 2),
                    "contract_value": value,
                    "submission_timestamp": burst + pd.Timedelta(seconds=int(rng.integers(0, 20 * 60))),
                })

            # a few genuine competitors who lose because the cartel controls the price signal
            n_out = int(rng.integers(0, 3))
            for o in rng.choice(outsiders_pool, size=n_out, replace=False):
                bids.append({
                    "contract_key": key, "company_id": o,
                    "bid_amount": round(win_amount * rng.uniform(1.03, 1.20), 2),
                    "contract_value": value,
                    "submission_timestamp": pub + (deadline - pub) * rng.uniform(0.1, 0.95),
                })
    return bids, contracts


def normal_bids(companies: pd.DataFrame, n_bids_needed: int, start_idx: int) -> tuple[list[dict], list[dict]]:
    bids, contracts = [], []

    # bidder counts that sum exactly to n_bids_needed
    counts, total = [], 0
    while total < n_bids_needed:
        c = int(min(1 + rng.poisson(3.0), 10, n_bids_needed - total))
        counts.append(c)
        total += c

    for j, n_bidders in enumerate(counts):
        key = f"NORMAL_{start_idx + j}"
        sector = rng.choice(SECTORS, p=SECTOR_WEIGHTS)
        value = contract_value(sector)
        pub, deadline = tender_window()
        contracts.append({"contract_key": key, "publication_date": pub,
                          "sector": sector, "is_rigged": 0, "cartel_id": pd.NA})

        same_sector = companies.loc[companies["sector"] == sector, "company_id"].to_numpy()
        # 90% of bidders from the contract's sector, the rest from anywhere
        n_same = min(len(same_sector), int(np.ceil(n_bidders * 0.9)))
        chosen = list(rng.choice(same_sector, size=n_same, replace=False))
        others = companies.loc[~companies["company_id"].isin(chosen), "company_id"].to_numpy()
        chosen += list(rng.choice(others, size=n_bidders - n_same, replace=False))

        for cid in chosen:
            ratio = float(np.clip(rng.normal(0.92, 0.08), 0.6, 1.3))   # independent pricing
            bids.append({
                "contract_key": key, "company_id": cid,
                "bid_amount": round(value * ratio, 2), "contract_value": value,
                "submission_timestamp": pub + (deadline - pub) * rng.uniform(0.05, 1.0),
            })
    return bids, contracts


def assemble(companies: pd.DataFrame, cartels: dict[str, list[str]]):
    c_bids, c_contracts = cartel_bids(companies, cartels)
    n_bids, n_contracts = normal_bids(companies, N_BIDS - len(c_bids), start_idx=0)

    contracts = pd.DataFrame(c_contracts + n_contracts).sort_values("publication_date")
    contracts["contract_id"] = [f"CTR{i:05d}" for i in range(1, len(contracts) + 1)]
    key_map = contracts.set_index("contract_key")["contract_id"]

    bids = pd.DataFrame(c_bids + n_bids)
    bids["contract_id"] = bids["contract_key"].map(key_map)
    bids["submission_timestamp"] = pd.to_datetime(bids["submission_timestamp"]).dt.floor("s")

    # lowest bid wins
    bids["winning_status"] = 0
    win_idx = bids.groupby("contract_id")["bid_amount"].idxmin()
    bids.loc[win_idx, "winning_status"] = 1

    bids = bids.sort_values("submission_timestamp").reset_index(drop=True)
    bids["bid_id"] = [f"BID{i:05d}" for i in range(1, len(bids) + 1)]
    bids = bids[["bid_id", "contract_id", "company_id", "bid_amount",
                 "winning_status", "submission_timestamp", "contract_value"]]

    gt_contracts = contracts[["contract_id", "sector", "publication_date", "is_rigged", "cartel_id"]]
    return bids, gt_contracts


# ---------------------------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------------------------
def validate(companies: pd.DataFrame, bids: pd.DataFrame, gt_contracts: pd.DataFrame) -> None:
    assert len(companies) == N_COMPANIES, "company count mismatch"
    assert len(bids) == N_BIDS, f"bid count mismatch: {len(bids)}"
    assert companies["company_id"].is_unique and bids["bid_id"].is_unique
    assert bids["company_id"].isin(companies["company_id"]).all(), "orphan company_id in bids"
    assert not bids.duplicated(["contract_id", "company_id"]).any(), "company bid twice on a contract"
    assert (bids.groupby("contract_id")["winning_status"].sum() == 1).all(), "need one winner per contract"
    assert (bids["bid_amount"] > 0).all()

    rigged = set(gt_contracts.loc[gt_contracts["is_rigged"] == 1, "contract_id"])
    cv = bids.groupby("contract_id")["bid_amount"].agg(lambda s: s.std() / s.mean() if len(s) > 1 else np.nan)
    print("\nValidation passed.")
    print(f"  contracts: {bids['contract_id'].nunique():,} ({len(rigged)} rigged)")
    print(f"  bids on rigged contracts: {bids['contract_id'].isin(rigged).sum():,}")
    print(f"  risk_flag = 1: {companies['risk_flag'].sum()} of {len(companies)}")
    print(f"  median bid CV, rigged contracts: {cv[cv.index.isin(rigged)].median():.4f}")
    print(f"  median bid CV, normal contracts: {cv[~cv.index.isin(rigged)].median():.4f}")


def check_owner_overlap(companies: pd.DataFrame) -> None:
    """Owner group size must not separate cartels from honest groups."""
    groups = companies.groupby("ultimate_beneficial_owner_id").agg(
        size=("company_id", "count"), is_cartel=("cartel_id", lambda s: s.notna().any()))
    groups = groups[groups["size"] > 1]
    table = pd.crosstab(groups["size"], groups["is_cartel"].map({True: "cartel", False: "honest"}))
    print("\nMulti-firm owners by group size:")
    print(table.to_string())
    cartel_sizes = set(groups.loc[groups["is_cartel"], "size"])
    honest_sizes = set(groups.loc[~groups["is_cartel"], "size"])
    assert len(cartel_sizes & honest_sizes) >= 3, "owner group sizes should overlap"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    companies = build_companies()
    cartels = plant_cartels(companies)
    add_noise(companies)
    bids, gt_contracts = assemble(companies, cartels)
    validate(companies, bids, gt_contracts)
    check_owner_overlap(companies)

    os.makedirs(OUT_DIR, exist_ok=True)
    gt_companies = companies[["company_id", "ultimate_beneficial_owner_id", "cartel_id", "risk_flag"]]
    companies_df = companies.drop(columns="cartel_id")
    companies_df["registration_date"] = companies_df["registration_date"].dt.date

    companies_df.to_csv(os.path.join(OUT_DIR, "companies.csv"), index=False)
    bids.to_csv(os.path.join(OUT_DIR, "bids.csv"), index=False)
    gt_companies.to_csv(os.path.join(OUT_DIR, "ground_truth_companies.csv"), index=False)
    gt_contracts.to_csv(os.path.join(OUT_DIR, "ground_truth_contracts.csv"), index=False)

    print("\nCartels planted:")
    for cid, members in cartels.items():
        print(f"  {cid}: {len(members)} members -> {', '.join(members)}")
    print(f"\nFiles written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
