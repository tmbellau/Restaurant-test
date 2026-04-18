# GLA People Counts — Manual Download

The London Datastore hosts the GLA "People Counts" dataset (O2 Motion mobile
data, hourly, by MSOA) under the Open Government Licence. Cloudflare blocks
automated downloads, so the CSV files must be obtained manually.

## Steps

1. Open https://data.london.gov.uk/dataset/busyness-people-counts in a browser
2. Scroll to the "Resources" section and click **Download** on each CSV file
3. Save them into this directory (`data/gla_busyness/`)
4. The pipeline auto-discovers all `*.csv` files here and filters to the
   target MSOA (default: `E02000972` = Fitzrovia West & Soho)

## Expected file schema

The pipeline handles variant column names, but you should see something like:

| column | example | meaning |
|--------|---------|---------|
| msoa_code or msoa11cd | E02000972 | 2011 MSOA code |
| datetime or timestamp | 2023-06-15 18:00 | Local hour |
| count or people_count | 4217 | People present in MSOA during hour |

## Target MSOA

- **Code:** `E02000972`
- **Name:** Fitzrovia West & Soho
- **Contains:** Wagamama Soho, 51.5131°N, -0.1318°W

## Licence

Open Government Licence v3.0 (OGL v3). Free to use with attribution:

> Contains data derived from O2 Motion, published by the Greater London
> Authority under the Open Government Licence v3.0.
