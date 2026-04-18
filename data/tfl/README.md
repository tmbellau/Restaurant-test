# TfL Daily Station Entry/Exit Data

Place TfL station entry/exit CSV files here. The pipeline filters to
**Tottenham Court Road** station (~400m from Wagamama Soho).

## Where to get the data

1. **TfL transparency page** — search for "TAPS daily rail station entry exit"
   on the TfL website's FOI section. Covers 2019-2023+.

2. **Kaggle** — https://www.kaggle.com/datasets/olisao/transport-for-london-tfl-entry-and-exit-dataset

3. **TfL Open Data** — https://tfl.gov.uk/info-for/open-data-users/our-open-data

## Expected format

The loader handles multiple CSV column naming conventions. At minimum:

| column | examples | meaning |
|--------|----------|---------|
| station / station_name | Tottenham Court Road | Station name |
| date / Date | 2023-06-15 | Calendar date |
| entries / Entries | 45230 | Daily entry taps |
| exits / Exits | 42180 | Daily exit taps |

## Licence

TfL Open Data — Open Government Licence v2.0 (OGLv2).
