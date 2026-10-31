# BANKING77 final L2 naming changes

Source: [`banking77_taxonomy_refined.csv`](banking77_taxonomy_refined.csv).  
Result: [`banking77_taxonomy_final.csv`](banking77_taxonomy_final.csv).

Only two L2 names changed. No L3 was moved, no L1 changed, and no rationale was edited.

## 1. `card_issuance` → `card_lifecycle_and_issuance`

The family covers ordering, delivery, activation, linking, spare cards, and expiry replacement. `card_issuance` named only the first step. `card_lifecycle_and_issuance` names the whole path from order to a usable card.

L1 stays `card`. Eight L3 labels:

- `activate_my_card`
- `card_about_to_expire`
- `card_arrival`
- `card_delivery_estimate`
- `card_linking`
- `get_physical_card`
- `getting_spare_card`
- `order_physical_card`

## 2. `withdrawal_discrepancy` → `withdrawal_issue`

The family is not only a wrong cash amount. It also covers an unrecognised ATM withdrawal and a wrong exchange rate on a withdrawal. `withdrawal_issue` covers those ATM-withdrawal problems without implying that every case is an amount mismatch.

L1 stays `cash_withdrawal`. Three L3 labels:

- `cash_withdrawal_not_recognised`
- `wrong_amount_of_cash_received`
- `wrong_exchange_rate_for_cash_withdrawal`

## Counts

| Rename | Rows |
|--------|-----:|
| `card_lifecycle_and_issuance` | 8 |
| `withdrawal_issue` | 3 |
| Modified rows | 11 |
| Unchanged rows | 66 |

L1 remains 8. L2 remains 23. L3 remains 77.
