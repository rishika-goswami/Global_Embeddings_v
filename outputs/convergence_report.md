# Convergence Report — Clustering of MORLY Grievances

**Sample size:** 15581 rows

## K found by each method × embedding

| Run | K |
|---|---|
| ymeans2/tfidf | 15 |
| ymeans2/sbert | 15 |
| dbscan/tfidf | 13 |
| dbscan/sbert | 21 |
| agg/tfidf | 4 |
| agg/sbert | 11 |

## Method summary (silhouette, davies-bouldin, noise%)

| method   | embedding   |   k |   noise_pct |   silhouette |   davies_bouldin |
|:---------|:------------|----:|------------:|-------------:|-----------------:|
| ymeans2  | tfidf       |  15 |        0    |       0.0817 |           4.4621 |
| ymeans2  | sbert       |  15 |        0    |       0.2245 |           2.2207 |
| dbscan   | tfidf       |  13 |       16.39 |      -0.0296 |           2.0813 |
| dbscan   | sbert       |  21 |        6.51 |      -0.1973 |           1.3787 |
| agg      | tfidf       |   4 |        0    |       0.0158 |           0.9859 |
| agg      | sbert       |  11 |        0    |       0.0983 |           0.8055 |

## Convergence assessment

- K range across methods: **4 – 21** (spread = 17)
- Recommended K (median): **14**
- ❌ High spread (17); the data does not strongly support a single K.

## Cross-method ARI (higher = more agreement)

|               |   ymeans2/tfidf |   ymeans2/sbert |   dbscan/tfidf |   dbscan/sbert |   agg/tfidf |   agg/sbert |
|:--------------|----------------:|----------------:|---------------:|---------------:|------------:|------------:|
| ymeans2/tfidf |            1    |            0.27 |          -0.02 |          -0.01 |          -0 |       -0    |
| ymeans2/sbert |            0.27 |            1    |          -0    |           0    |          -0 |        0    |
| dbscan/tfidf  |           -0.02 |           -0    |           1    |           0.12 |           0 |        0    |
| dbscan/sbert  |           -0.01 |            0    |           0.12 |           1    |          -0 |        0.02 |
| agg/tfidf     |           -0    |           -0    |           0    |          -0    |           1 |       -0    |
| agg/sbert     |           -0    |            0    |           0    |           0.02 |          -0 |        1    |

## Cross-method NMI

|               |   ymeans2/tfidf |   ymeans2/sbert |   dbscan/tfidf |   dbscan/sbert |   agg/tfidf |   agg/sbert |
|:--------------|----------------:|----------------:|---------------:|---------------:|------------:|------------:|
| ymeans2/tfidf |            1    |            0.51 |           0.04 |           0.02 |           0 |        0    |
| ymeans2/sbert |            0.51 |            1    |           0.03 |           0.02 |           0 |        0    |
| dbscan/tfidf  |            0.04 |            0.03 |           1    |           0.04 |           0 |        0    |
| dbscan/sbert  |            0.02 |            0.02 |           0.04 |           1    |           0 |        0.01 |
| agg/tfidf     |            0    |            0    |           0    |           0    |           1 |        0    |
| agg/sbert     |            0    |            0    |           0    |           0.01 |           0 |        1    |

## Top keywords per cluster (per run)

### ymeans2/tfidf

- **Class 0**: bahut kharab, kharab, bahut, station, safai, safai bahut, platform, station safai
- **Class 1**: behavior, staff behavior, staff, behavior bahut, behavior theek, theek nahi, theek, nahi
- **Class 2**: safai, safai nahi, ganda, bahut ganda, platform, nahi bahut, bahut, platform safai
- **Class 3**: refund nahi, refund, nahi mila, mila, nahi, ticket, cancel, ticket refund
- **Class 4**: payment, payment nahi, ticket, nahi, book, ticket book, ticket payment, book payment
- **Class 5**: train, bahut, nahi, nahi train, late, platform, dikkat, delay
- **Class 6**: paisa, ticket, paisa nahi, ticket paisa, nahi, mila, nahi mila, wapas
- **Class 7**: time, train time, time nahi, train, nahi, bahut, time schedule, late
- **Class 8**: staff, training, nahi, rude, staff training, train staff, railway staff, bahut
- **Class 9**: confirmation, confirmation nahi, payment confirmation, payment, nahi mila, mila, nahi, ticket payment
- **Class 10**: timing, train timing, train, timing bahut, late, timing sahi, bahut, nahi
- **Class 11**: facilities, station facilities, facilities bahut, station, facilities nahi, kharab, bahut, bahut kharab
- **Class 12**: train schedule, schedule, train, schedule bahut, bahut, nahi, nahi train, badal
- **Class 13**: baar, baar baar, train, train baar, nahi baar, baar late, baar change, late
- **Class 14**: station, railway station, bahut, railway, building, station building, purani, station bahut

### ymeans2/sbert

- **Class 0**: ticket, payment, payment nahi, ticket payment, nahi, book, ticket book, nahi ticket
- **Class 1**: train, nahi, bahut, nahi train, time nahi, time, train time, late
- **Class 2**: staff, train staff, railway staff, railway, behavior, train, training, nahi
- **Class 3**: platform, bahut, platform bahut, safai, safai nahi, kharab, safety, bahut kharab
- **Class 4**: refund, refund nahi, nahi mila, mila, nahi, nahi refund, refund process, process
- **Class 5**: payment, confirmation, confirmation nahi, nahi, payment confirmation, nahi mila, mila, payment nahi
- **Class 6**: station, bahut, washroom, toilets, gande, kharab, station washroom, facilities
- **Class 7**: train, schedule, train schedule, timing, train timing, bahut, late, time
- **Class 8**: bahut, train, dikkat, bahut dikkat, station, railway, chahiye, nahi
- **Class 9**: refund, refund nahi, ticket, nahi mila, mila, cancel, ticket refund, ticket cancel
- **Class 10**: baar, baar baar, train, train baar, baar change, baar late, late, change
- **Class 11**: paisa, ticket, ticket paisa, paisa nahi, nahi, wapas, mila, nahi mila
- **Class 12**: safai, nahi, bahut, safai nahi, platform, station, nahi bahut, station safai
- **Class 13**: station, railway station, railway, bahut, kharab, facilities, infrastructure, bahut kharab
- **Class 14**: staff, training, behavior, nahi, staff behavior, staff training, zarurat, bahut

### dbscan/tfidf

- **Class 0**: nahi, bahut, train, station, staff, ticket, payment, baar
- **Class 1**: train seats, seats, train, dikkat, nahi, booking, confirm nahi, confirm
- **Class 2**: bahut garmi, garmi, kaam nahi, kaam, train, nahi bahut, train nahi, bahut
- **Class 3**: platforms bahut, platforms, bahut, kharab, bahut kharab, zaroorat, sakti, samasya bahut
- **Class 4**: samadhan chahiye, samadhan, chahiye, payment issue, issue, train der, clarity, charge
- **Class 5**: tracks, railway tracks, railway, kharab, halat, condition, bahut kharab, bahut
- **Class 6**: support nahi, support, nahi mila, mila, nahi, staff, nahi nahi, training
- **Class 7**: staff log, log, rude, bahut rude, staff, bahut, samjha, rude nahi
- **Class 8**: bahut slow, slow, service bahut, service, bahut, train service, improvement nahi, improvement
- **Class 9**: iski jaanch, jaanch, iski, pehle, der, refund, nahi mila, mila
- **Class 10**: counter staff, ticket counter, counter, ticket, staff, staff nahi, accha nahi, accha
- **Class 11**: train services, services, train, sudhar, bahut pareshani, bahut, pareshani, samasya
- **Class 12**: achhi nahi, achhi, staff training, training, staff, train staff, service, unse

### dbscan/sbert

- **Class 0**: nahi, bahut, train, staff, station, ticket, payment, baar
- **Class 1**: nahi paisa, paisa, ticket, ticket payment, karu, payment nahi, karun, payment
- **Class 2**: bahut slow, slow, response, staff, time bahut, time, bahut, chahiye
- **Class 3**: confirm nahi, confirm, seat confirm, train seat, seat, nahi bahut, train seats, seats
- **Class 4**: train guard, guard, main bahut, train, khud, bahut pareshan, bura laga, pareshan
- **Class 5**: quality, train, khana, service, service nahi, nahi, bohot, bahut
- **Class 6**: train bahut, arrangement, bahut, nahi train, travel, train, trains, seat
- **Class 7**: nahi bahut, staff, bahut bekaar, achhe nahi, bekaar, bahut problem, experience, achhe
- **Class 8**: kharab logon, nahi train, train platform, nahi log, train, train bahut, bahut mushkil, logon
- **Class 9**: band, station lift, lift, station, dikkat jaldi, mushkil, jaldi theek, jaldi
- **Class 10**: platform, dikkat, bahut dikkat, bahut, platform bahut
- **Class 11**: repair, repair zarurat, zarurat, bahut zaruri, nahi wajah, vyavastha, zaruri, bahut kharab
- **Class 12**: vyavhaar, bura vyavhaar, saath, action, saath staff, bura, staff, staff saath
- **Class 13**: complaint, ignore, staff, bahut galat, dhyaan, bura laga, ispar, laga
- **Class 14**: charge, ticket, bahut galat, amount, ticket book, book, check, galat
- **Class 15**: bahut kharab, aaj, kharab, condition, bahut, purani, zaroori, kaam
- **Class 16**: seating, station seating, seats, area, issue, check, train, railway station
- **Class 17**: payment process, process, delay, payment, delays, number xxxxxx, account number, number
- **Class 18**: condition bahut, condition, railway, bahut, bahut purana, bahut kharab, maintenance, purana
- **Class 19**: kafi, payment issue, din, issue, payment, mila payment, payment status, status
- **Class 20**: cancel, cancel paisa, paisa, paisa nahi, mila karun, nahi mila, mila, bahut samay

### agg/tfidf

- **Class 0**: nahi, bahut, train, staff, station, ticket, payment, baar
- **Class 1**: employees, aksar, apne, der, kaam, karte, samasya, train
- **Class 2**: isme, padega, abhi, late, train
- **Class 3**: jaati, bar, har, nahi sakte, ghante, der, sakte, train

### agg/sbert

- **Class 0**: nahi, bahut, train, staff, station, ticket, payment, baar
- **Class 1**: reservation, system, thoda, chahiye, bahut
- **Class 2**: aaye, check
- **Class 3**: action, platform
- **Class 4**: jaldi theek, main, jaldi, theek
- **Class 5**: achi nahi, achi, platform, dikkat, nahi
- **Class 6**: chahiye bahut, bahut zaroori, zaroori, chahiye, station, bahut
- **Class 7**: unacceptable
- **Class 8**: quality, dhyan dein, dein, customer service, customer, bekar, service, dhyan
- **Class 9**: samjhana, guard, padega, bahut rude, rude, bahut
- **Class 10**: bahut samasya, samasya, platform, bahut

## Files produced

- `agglomerative_param_grid_sbert.png`
- `agglomerative_param_grid_tfidf.png`
- `agglomerative_sweep_sbert.csv`
- `agglomerative_sweep_tfidf.csv`
- `cluster_keywords.json`
- `cluster_size_distribution.png`
- `convergence_report.md`
- `convergence_summary.png`
- `cosine_similarity_heatmap.png`
- `cross_method_ari.csv`
- `cross_method_ari_heatmap.png`
- `cross_method_nmi.csv`
- `dbscan_k_distance_sbert.png`
- `dbscan_k_distance_tfidf.png`
- `dbscan_param_grid_sbert.png`
- `dbscan_param_grid_tfidf.png`
- `dbscan_sweep_sbert.csv`
- `dbscan_sweep_tfidf.csv`
- `dendrogram_sbert_all_linkages.png`
- `dendrogram_sbert_chosen.png`
- `dendrogram_tfidf_all_linkages.png`
- `dendrogram_tfidf_chosen.png`
- `llm_taxonomy.json`
- `method_summary.csv`
- `sbert_cache_15581.npy`
- `sbert_cache_15582.npy`
- `sbert_cache_5000.npy`
- `ymeans2_sweep_sbert.csv`
- `ymeans2_sweep_tfidf.csv`
- `ymeans_bic_curve.png`