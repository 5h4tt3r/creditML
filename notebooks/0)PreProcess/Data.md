## Credit Card Transactions Dataset — Column Reference

| Column                    | Description                                                    | Primary Use Cases                                      | Feature Engineering Ideas                                                                                          |
|---------------------------|----------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **trans_date_trans_time** | Full timestamp of the transaction (e.g. “2019-01-01 00:00:18”) | • Time-series forecasting<br>• Hour/day‐of‐week analysis | • Extract `hour`, `weekday`, `month`, `year`<br>• Flag weekends/holidays<br>• Rolling statistics per card           |
| **cc_num**                | Anonymized credit‐card number                                  | • Grouping by cardholder<br>• Per‐user behavior trends  | • Count txns per card over windows<br>• Time since last txn<br>• Fraud‐rate per card as historical feature         |
| **merchant**              | Merchant or store name                                         | • Merchant‐level fraud rates<br>• Popularity analysis    | • Frequency / recency counts<br>• Target‐encode or one‐hot encode<br>• Text‐hash embedding                          |
| **category**              | Transaction category (e.g. grocery, entertainment)             | • Behavior segmentation<br>• Category‐specific fraud     | • One‐hot / ordinal encode<br>• Aggregate spend per category<br>• Category transition sequences                     |
| **amt**                   | Transaction amount (in USD)                                    | • Anomaly detection<br>• Spend forecasting               | • Log or Box–Cox transform<br>• Binning into tiers<br>• Rolling sum/mean amounts                                    |
| **first**, **last**       | Cardholder’s first and last name                              | • (Lookup only; usually dropped)                        | • Combine as synthetic `user_id`<br>• Drop for modeling to preserve privacy                                        |
| **gender**                | Cardholder gender (`M` / `F`)                                  | • Demographic segmentation                             | • Binary encoding (0/1)<br>• Interaction with age groups                                                           |
| **street**, **city**, **state**, **zip** | Cardholder address fields                       | • Home‐location profiling<br>• Geo‐anomaly detection     | • Geocode to lat/long<br>• Distance from home→merchant<br>• Urban/rural flag via ZIP                                  |
| **lat**, **long**         | Cardholder’s geo‐coordinates                                   | • Geo‐clustering<br>• Location‐based fraud detection     | • Cluster into regions or grids<br>• Distance to merchant location                                                 |
| **city_pop**              | Population of the cardholder’s city                            | • Contextual fraud risk<br>• Behavior modeling           | • Log‐scale or bucket into size‐classes (small/medium/large)                                                       |
| **job**                   | Cardholder occupation                                          | • Socio‐economic profiling                              | • Group rare jobs into “Other”<br>• Target‐encode or frequency encode                                                |
| **dob**                   | Cardholder date of birth (YYYY-MM-DD)                          | • Age calculation                                       | • Derive `age` at txn time<br>• Bucket into age‐bands                                                           |
| **trans_num**             | Unique transaction identifier                                  | • (Lookup only)                                         | • Drop or use for indexing; rarely used as model feature                                                           |
| **unix_time**             | Unix epoch timestamp of txn                                     | • Alternative time representation                       | • Direct time‐based features<br>• Difference in seconds between txns                                                |
| **merch_lat**, **merch_long** | Merchant geo‐coordinates                                  | • Merchant location profiling<br>• Geo‐distance checks   | • Compute distance from cardholder home<br>• Region clustering                                                      |
| **merch_zipcode**         | Merchant ZIP code                                              | • Merchant location profiling                            | • Compare with user ZIP for distance anomalies<br>• One‐hot or frequency encode                                      |
| **is_fraud**              | Fraud label (1 = fraud, 0 = legitimate)                         | • **Target** for fraud‐detection classifiers            | —                                                                                                                  |

---

### Additional Notes
- **Drop:** `first`, `last`, `trans_num` (unless needed for logging/indexing).
- **Datetime engineering:** Extract granular features from `trans_date_trans_time` for rich time‐based signals.
- **Aggregation:** For each `cc_num` compute windowed statistics (count, sum, fraud‐rate, avg amt).
- **Geo‐features:** Combine `lat/long` (cardholder) with `merch_lat/long` to derive travel distance, speed, and region anomalies.
- **High‐cardinality encoding:** Use target or frequency encoding for `merchant`, `job`, and `zip` to control dimensionality.
