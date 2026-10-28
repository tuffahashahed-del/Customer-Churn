# Data Dictionary

Prediction time: **the beginning of the current month**. Target: whether the customer leaves in **the following month**. Rows describe simulated customers, not real people.

| Column | Meaning | Notes |
| --- | --- | --- |
| `customer_id` | Customer record identifier | An ID, not a behavioral feature; some training rows may be repeated |
| `tenure_months` | Months with the provider | Numeric, should not be negative |
| `monthly_bill` | Current monthly charge, in synthetic currency units | Numeric |
| `support_tickets_90d` | Support requests over prior 90 days | Numeric count |
| `late_payments_6m` | Late payments over prior six months | Numeric count |
| `contract_type` | Current contract length/type | Categorical |
| `auto_pay` | Auto-pay enrollment | Categorical |
| `internet_type` | Main internet access type | Categorical |
| `weekly_usage_gb` | Recent average weekly usage | Numeric, should not be negative |
| `region` | Service region | Categorical |
| `case_status` | Administrative outcome for a customer service case, which **may be updated after the prediction point** | Consider feature availability, potential leakage, and appropriate exclusion |
| `churned_next_month` | Label: customer churned next month | 1 = Yes, 0 = No; **present only in train** |

Values may be missing, inconsistently formatted, duplicated, or implausible. It is your responsibility to investigate and document how you handled each issue.
