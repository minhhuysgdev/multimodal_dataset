# BANKING77 taxonomy refinement log

Baseline: [`banking77_hierarchy.csv`](banking77_hierarchy.csv).  
Refined file: [`banking77_taxonomy_refined.csv`](banking77_taxonomy_refined.csv).  
The baseline file was not overwritten.

Scope: targeted edits only. L1 domains are unchanged. L3 label IDs 0–76 are unchanged.

## Applied changes

### Transfer: split `transfer_setup`

`transfer_setup` mixed payee setup with money arriving. Outgoing lifecycle stays in `transfer_status`.

| label_id | L3 | Before | After | Reason |
|----------|----|--------|-------|--------|
| 7 | `beneficiary_not_allowed` | `transfer` / `transfer_setup` | `transfer` / `transfer_configuration` | The customer cannot add or use a beneficiary. That is payee configuration, not an incoming credit and not an outgoing status. |
| 50 | `receiving_money` | `transfer` / `transfer_setup` | `transfer` / `incoming_transfer` | The customer is waiting for money sent by someone else. |
| 65 | `transfer_into_account` | `transfer` / `transfer_setup` | `transfer` / `incoming_transfer` | The customer asks about a transfer posted into their own account. |

Unchanged under `transfer` / `transfer_status`: `cancel_transfer`, `declined_transfer`, `failed_transfer`, `pending_transfer`, `transfer_not_received_by_recipient`.

### Top-up: separate verification from funding method

| label_id | L3 | Before | After | Reason |
|----------|----|--------|-------|--------|
| 71 | `verify_top_up` | `top_up` / `top_up_method` | `top_up` / `top_up_verification` | Utterances ask how to verify the card used to add money. That is not a funding method and not identity KYC (`verify_my_identity`, `why_verify_identity`, `unable_to_verify_identity`, `verify_source_of_funds`). |

Unchanged under `top_up_method`: `automatic_top_up`, `top_up_by_cash_or_cheque`, `topping_up_by_card`, `supported_cards_and_currencies`.

### Charges: rename the dispute family

L1 stays `refund_and_charges`. Child rationales are unchanged because the L3 meaning did not change.

| label_id | L3 | Before L2 | After L2 | Reason |
|----------|----|-----------|----------|--------|
| 28 | `direct_debit_payment_not_recognised` | `unrecognised_or_extra_charge` | `charge_dispute` | Customer does not recognise a direct debit. |
| 34 | `extra_charge_on_statement` | `unrecognised_or_extra_charge` | `charge_dispute` | Customer disputes an extra statement charge. |
| 63 | `transaction_charged_twice` | `unrecognised_or_extra_charge` | `charge_dispute` | Customer disputes a duplicate charge. |

`charge_dispute` names the customer problem (a charge they contest) more directly than “unrecognised or extra”.

Not moved into this family: `cash_withdrawal_charge` (ATM channel) and `card_payment_fee_charged` (card payment).

### L3 names

The two irregular Hugging Face strings stay exactly as published. A later spelling cleanup was reverted.

| label_id | L3 kept | Path |
|----------|---------|------|
| 51 | `Refund_not_showing_up` | `refund_and_charges` / `refund` |
| 53 | `reverted_card_payment?` | `card` / `card_payment` |

## Reviewed and left unchanged

### Card issuance

`card_issuance` still holds ordering, delivery, activation, linking, spare card, and expiry replacement:

`order_physical_card`, `get_physical_card`, `getting_spare_card`, `card_about_to_expire`, `card_arrival`, `card_delivery_estimate`, `activate_my_card`, `card_linking`.

Rejected: four new L2s for acquisition, delivery, activation, and lifecycle. Those slices are real, but each would be a thin label for steps of the same “get a physical card into use” journey. A separate `card_lifecycle` only for expiry and spare cards would also split replacement away from `order_physical_card` without a cleaner boundary. Keeping one L2 is the more consistent assignment.

### Currency exchange

`exchange_info` still holds `exchange_charge`, `exchange_rate`, `exchange_via_app`, and `fiat_currency_support`. They are questions about exchanging money or which fiat currencies the app supports. Splitting the fee from the rate would recreate a one-child family.

### ATM channel

`atm_operation` still holds `atm_support`, `card_swallowed`, `cash_withdrawal_charge`, `declined_cash_withdrawal`, and `pending_cash_withdrawal`.

`withdrawal_discrepancy` still holds `cash_withdrawal_not_recognised`, `wrong_amount_of_cash_received`, and `wrong_exchange_rate_for_cash_withdrawal`.

The split is by customer problem: operating the ATM versus the cash or rate not matching what the customer expected. Withdrawal fees stay with the ATM channel and are not treated as a statement dispute.

## Resulting size

| | Baseline | Refined |
|--|----------|---------|
| L1 | 8 | 8 |
| L2 | 21 | 23 |
| L3 | 77 | 77 |

New L2 names: `transfer_configuration`, `incoming_transfer`, `top_up_verification`.  
Removed L2 names: `transfer_setup`, `unrecognised_or_extra_charge` (renamed to `charge_dispute`).
