# Procurement Collusion Detection

Detecting bid-rigging networks in public procurement with graph analytics and anomaly detection.

> Work in progress. This README grows as each stage of the project is completed.

## The problem

Bid rigging is hard to catch one contract at a time. A single tender with four bids that are close in price looks normal. The pattern only shows up when you look across many tenders at once: the same group of firms showing up together again and again, pricing within a fraction of a percent of each other, taking turns to win, and often tracing back to the same owner.

This project treats procurement as a network of companies, contracts and owners, and asks whether that structure can surface collusive groups that contract-level analysis would miss.

## Data

The project starts on a synthetic dataset where the collusion is planted on purpose, so the methods can be checked against a known answer before they are applied to real open contracting data.

| File | Rows | Description |
|---|---|---|
| `data/companies.csv` | 500 | Company profile, registration date, beneficial owner, sector, known risk flag |
| `data/bids.csv` | 5,000 | Bids across 1,240 contracts: amount, outcome, timestamp, contract value |
| `data/ground_truth_companies.csv` | 500 | Cartel membership. **Used for evaluation only** |
| `data/ground_truth_contracts.csv` | 1,240 | Which contracts were rigged. **Used for evaluation only** |

The generator (`src/generate_procurement_data.py`, seed 42) plants six cartels. Members share a beneficial owner, were registered within weeks of each other, bid together on every target contract with near-identical prices, submit within minutes of each other and rotate the win.

To keep the task honest, it also includes:

- **Partial labels.** Only some cartel members carry `risk_flag = 1`, and a few innocent firms are wrongly flagged.
- **Decoys.** Twelve legitimate owners hold two firms each that never collude, so shared ownership alone is not enough.
- **Mixed behaviour.** Cartel members also bid normally on ordinary tenders.

## Project stages

- [ ] 1. Graph construction and exploration
- [ ] 2. Co-bidding network and bid similarity
- [ ] 3. Red-flag feature engineering
- [ ] 4. Community detection
- [ ] 5. Anomaly scoring
- [ ] 6. Explanation and case studies
- [ ] 7. Application to real OCDS procurement data

## Repository structure

```
├── data/        synthetic datasets
├── notebooks/   analysis notebooks, one per stage
├── src/         data generator and reusable code
└── README.md
```

## Reproducing

```bash
pip install -r requirements.txt
python src/generate_procurement_data.py
```

## Author

Prince Amofa, MSc Data Science, University of Ghana
