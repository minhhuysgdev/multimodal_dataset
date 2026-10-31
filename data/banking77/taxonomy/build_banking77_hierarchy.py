#!/usr/bin/env python3
"""Generate BANKING77 hierarchical taxonomy artifacts (L1/L2/L3)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Official Hugging Face label_id -> L3 (original intent name)
HF_L3 = {
    0: "activate_my_card",
    1: "age_limit",
    2: "apple_pay_or_google_pay",
    3: "atm_support",
    4: "automatic_top_up",
    5: "balance_not_updated_after_bank_transfer",
    6: "balance_not_updated_after_cheque_or_cash_deposit",
    7: "beneficiary_not_allowed",
    8: "cancel_transfer",
    9: "card_about_to_expire",
    10: "card_acceptance",
    11: "card_arrival",
    12: "card_delivery_estimate",
    13: "card_linking",
    14: "card_not_working",
    15: "card_payment_fee_charged",
    16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate",
    18: "card_swallowed",
    19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised",
    21: "change_pin",
    22: "compromised_card",
    23: "contactless_not_working",
    24: "country_support",
    25: "declined_card_payment",
    26: "declined_cash_withdrawal",
    27: "declined_transfer",
    28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits",
    30: "edit_personal_details",
    31: "exchange_charge",
    32: "exchange_rate",
    33: "exchange_via_app",
    34: "extra_charge_on_statement",
    35: "failed_transfer",
    36: "fiat_currency_support",
    37: "get_disposable_virtual_card",
    38: "get_physical_card",
    39: "getting_spare_card",
    40: "getting_virtual_card",
    41: "lost_or_stolen_card",
    42: "lost_or_stolen_phone",
    43: "order_physical_card",
    44: "passcode_forgotten",
    45: "pending_card_payment",
    46: "pending_cash_withdrawal",
    47: "pending_top_up",
    48: "pending_transfer",
    49: "pin_blocked",
    50: "receiving_money",
    51: "Refund_not_showing_up",
    52: "request_refund",
    53: "reverted_card_payment?",
    54: "supported_cards_and_currencies",
    55: "terminate_account",
    56: "top_up_by_bank_transfer_charge",
    57: "top_up_by_card_charge",
    58: "top_up_by_cash_or_cheque",
    59: "top_up_failed",
    60: "top_up_limits",
    61: "top_up_reverted",
    62: "topping_up_by_card",
    63: "transaction_charged_twice",
    64: "transfer_fee_charged",
    65: "transfer_into_account",
    66: "transfer_not_received_by_recipient",
    67: "transfer_timing",
    68: "unable_to_verify_identity",
    69: "verify_my_identity",
    70: "verify_source_of_funds",
    71: "verify_top_up",
    72: "virtual_card_not_working",
    73: "visa_or_mastercard",
    74: "why_verify_identity",
    75: "wrong_amount_of_cash_received",
    76: "wrong_exchange_rate_for_cash_withdrawal",
}

# label_id -> (L1, L2, flag, rationale)
MAPPING: dict[int, tuple[str, str, str, str]] = {
    0: ("card", "card_issuance", "", "Post-delivery activation of physical card."),
    11: ("card", "card_issuance", "", "Tracking or delay of card delivery."),
    12: ("card", "card_issuance", "", "Expected delivery timeline for a card."),
    43: ("card", "card_issuance", "", "Ordering a new physical card."),
    38: ("card", "card_issuance", "", "How to obtain a physical card."),
    39: ("card", "card_issuance", "", "Replacement or spare physical card."),
    13: ("card", "card_issuance", "", "Linking card to wallet or account."),
    40: ("card", "virtual_card", "", "Creating or accessing a virtual card."),
    37: ("card", "virtual_card", "", "Disposable virtual card issuance."),
    29: ("card", "virtual_card", "", "Limits on disposable virtual cards."),
    72: ("card", "virtual_card", "", "Virtual card malfunction."),
    21: ("card", "card_security", "", "PIN change workflow."),
    49: ("card", "card_security", "", "PIN blocked after failed attempts."),
    22: ("card", "card_security", "", "Suspected fraud or compromised card."),
    41: ("card", "card_security", "", "Report lost or stolen card."),
    14: ("card", "card_function", "", "Card fails for payments or general use."),
    23: ("card", "card_function", "", "Contactless payment not working."),
    18: (
        "cash_withdrawal",
        "atm_operation",
        "",
        "ATM retained the card; same channel as cash withdrawal, not a card-function failure.",
    ),
    9: (
        "card",
        "card_issuance",
        "",
        "Card expiring and user asks for a replacement; issuance lifecycle, not a malfunction.",
    ),
    25: ("card", "card_payment", "", "Card payment declined."),
    45: ("card", "card_payment", "", "Card payment pending settlement."),
    53: ("card", "card_payment", "", "Card payment reverted or reversed."),
    16: ("card", "card_payment", "", "Unrecognised card payment on statement."),
    15: ("card", "card_payment", "", "Fee charged on card payment."),
    17: ("card", "card_payment", "", "Wrong FX rate applied to card payment."),
    10: ("card", "card_product_info", "", "Merchant or terminal card acceptance."),
    73: ("card", "card_product_info", "", "Visa vs Mastercard product question."),
    2: ("card", "card_product_info", "", "Apple Pay / Google Pay support."),
    54: (
        "top_up",
        "top_up_method",
        "",
        "Which cards or currencies can be used to add money; top-up instrument, not the user's card product.",
    ),
    3: ("cash_withdrawal", "atm_operation", "", "General ATM support."),
    46: ("cash_withdrawal", "atm_operation", "", "Cash withdrawal pending."),
    26: ("cash_withdrawal", "atm_operation", "", "Cash withdrawal declined."),
    20: ("cash_withdrawal", "withdrawal_discrepancy", "", "Unrecognised ATM withdrawal."),
    75: ("cash_withdrawal", "withdrawal_discrepancy", "", "Wrong cash amount dispensed."),
    19: (
        "cash_withdrawal",
        "atm_operation",
        "",
        "Fee charged when using an ATM; same channel as withdrawal, not a dispensed-amount discrepancy.",
    ),
    76: ("cash_withdrawal", "withdrawal_discrepancy", "", "Wrong FX on cash withdrawal."),
    7: ("transfer", "transfer_setup", "", "Beneficiary not allowed for transfer."),
    65: ("transfer", "transfer_setup", "", "Incoming transfer to own account."),
    50: ("transfer", "transfer_setup", "", "Receiving money from others."),
    48: ("transfer", "transfer_status", "", "Outgoing transfer pending."),
    35: ("transfer", "transfer_status", "", "Transfer failed."),
    27: ("transfer", "transfer_status", "", "Transfer declined."),
    8: ("transfer", "transfer_status", "", "Cancel an outgoing transfer."),
    66: ("transfer", "transfer_status", "", "Recipient did not receive transfer."),
    67: ("transfer", "transfer_info_and_fee", "", "How long transfers take."),
    64: ("transfer", "transfer_info_and_fee", "", "Fee charged on transfer."),
    4: ("top_up", "top_up_method", "", "Automatic top-up configuration."),
    62: ("top_up", "top_up_method", "", "Top up using another card."),
    58: ("top_up", "top_up_method", "", "Top up via cash or cheque."),
    47: ("top_up", "top_up_status", "", "Top-up pending."),
    59: ("top_up", "top_up_status", "", "Top-up failed."),
    61: ("top_up", "top_up_status", "", "Top-up reverted."),
    71: (
        "top_up",
        "top_up_method",
        "",
        "How to verify the card used for a top-up; not a posting status and not identity KYC.",
    ),
    5: (
        "top_up",
        "top_up_status",
        "",
        "Inbound bank transfer not reflected in balance; utterances are top-up posting, not an outgoing transfer.",
    ),
    6: ("top_up", "top_up_status", "", "Balance not updated after cash/cheque deposit."),
    56: ("top_up", "top_up_fee_and_limit", "", "Fee for top-up by bank transfer."),
    57: ("top_up", "top_up_fee_and_limit", "", "Fee for top-up by card."),
    60: ("top_up", "top_up_fee_and_limit", "", "Top-up amount limits."),
    32: ("currency_exchange", "exchange_info", "", "General exchange rate inquiry."),
    33: ("currency_exchange", "exchange_info", "", "Exchange via mobile app."),
    36: ("currency_exchange", "exchange_info", "", "Supported fiat currencies."),
    31: (
        "currency_exchange",
        "exchange_info",
        "",
        "Currency-exchange fee; same family as rate, in-app exchange, and fiat support.",
    ),
    52: ("refund_and_charges", "refund", "", "Customer requests a refund."),
    51: ("refund_and_charges", "refund", "", "Refund not visible on account."),
    28: ("refund_and_charges", "unrecognised_or_extra_charge", "", "Unrecognised direct debit."),
    34: ("refund_and_charges", "unrecognised_or_extra_charge", "", "Extra charge on statement."),
    63: ("refund_and_charges", "unrecognised_or_extra_charge", "", "Same transaction charged twice."),
    69: ("identity_verification", "kyc", "", "Complete identity verification."),
    74: ("identity_verification", "kyc", "", "Why identity verification is required."),
    68: ("identity_verification", "kyc", "", "Unable to complete identity verification."),
    70: ("identity_verification", "kyc", "", "Source-of-funds verification."),
    30: ("account_and_access", "account_lifecycle", "", "Edit personal profile details."),
    55: ("account_and_access", "account_lifecycle", "", "Close or terminate account."),
    1: ("account_and_access", "eligibility", "", "Minimum age or age limit policy."),
    24: ("account_and_access", "eligibility", "", "Country availability support."),
    44: ("account_and_access", "app_access_security", "", "Forgotten app passcode."),
    42: ("account_and_access", "app_access_security", "", "Lost or stolen phone with app access."),
}


def build_rows() -> list[dict]:
    rows = []
    for label_id in range(77):
        l3 = HF_L3[label_id]
        if label_id not in MAPPING:
            raise KeyError(f"Missing mapping for label_id={label_id} ({l3})")
        l1, l2, flag, rationale = MAPPING[label_id]
        if HF_L3[label_id] != l3:
            pass
        rows.append(
            {
                "label_id": label_id,
                "L3": l3,
                "L2": l2,
                "L1": l1,
                "flag": flag,
                "rationale": rationale,
            }
        )
    return rows


def build_tree(rows: list[dict]) -> dict:
    tree: dict = {"root": "BANKING", "children": {}}
    for row in rows:
        l1, l2, l3 = row["L1"], row["L2"], row["L3"]
        tree["children"].setdefault(l1, {"children": {}})
        tree["children"][l1]["children"].setdefault(l2, {"children": []})
        tree["children"][l1]["children"][l2]["children"].append(
            {"label_id": row["label_id"], "L3": l3}
        )
    for l1 in tree["children"]:
        for l2 in tree["children"][l1]["children"]:
            tree["children"][l1]["children"][l2]["children"].sort(
                key=lambda x: x["label_id"]
            )
    return tree


def audit(rows: list[dict]) -> dict:
    ids = [r["label_id"] for r in rows]
    l3_names = [r["L3"] for r in rows]
    paths = [(r["L1"], r["L2"], r["L3"]) for r in rows]
    l1_counts = Counter(r["L1"] for r in rows)
    l2_counts = Counter((r["L1"], r["L2"]) for r in rows)
    flagged = [r for r in rows if r["flag"]]
    return {
        "l3_count": len(rows),
        "unique_ids": len(set(ids)),
        "ids_complete_0_76": set(ids) == set(range(77)),
        "l3_match_hf": all(r["L3"] == HF_L3[r["label_id"]] for r in rows),
        "unique_paths": len(set(paths)),
        "duplicate_paths": len(paths) - len(set(paths)),
        "orphans": 0 if len(rows) == 77 and len(set(ids)) == 77 else "check_failed",
        "L1_counts": dict(sorted(l1_counts.items())),
        "L2_count": len(l2_counts),
        "flagged_count": len(flagged),
        "flagged_intents": [
            {"label_id": r["label_id"], "L3": r["L3"], "flag": r["flag"]} for r in flagged
        ],
    }


def main() -> None:
    rows = build_rows()
    audit_report = audit(rows)
    if not audit_report["ids_complete_0_76"] or audit_report["duplicate_paths"]:
        raise SystemExit(f"Audit failed: {audit_report}")

    csv_path = ROOT / "banking77_hierarchy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label_id", "L3", "L2", "L1", "flag", "rationale"]
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": "1.0",
        "dataset": "PolyAI/banking77",
        "convention": {
            "root": "BANKING (outside L1-L2-L3)",
            "L3": "Original Hugging Face intent name; unchanged semantics and identifier",
            "L1": "Primary banking product line",
            "L2": "Intent family within L1",
        },
        "coverage": {
            "l3": audit_report["l3_count"],
            "unique_ids": audit_report["unique_ids"],
            "orphans": 0,
            "label_ids": "0-76 inclusive",
        },
        "statistics": {
            "L1_counts": audit_report["L1_counts"],
            "L2_count": audit_report["L2_count"],
        },
        "flat": rows,
        "tree": build_tree(rows),
        "audit": audit_report,
        "citation": {
            "title": "Efficient Intent Detection with Dual Sentence Encoders",
            "authors": "Casanueva et al.",
            "year": 2020,
            "url": "https://huggingface.co/datasets/PolyAI/banking77",
        },
    }
    json_path = ROOT / "banking77_hierarchy.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    from generate_taxonomy_visuals import generate_all

    for path in generate_all(rows, audit_report):
        print("Wrote", path)

    print("Wrote", csv_path)
    print("Wrote", json_path)
    print("Audit:", json.dumps(audit_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
