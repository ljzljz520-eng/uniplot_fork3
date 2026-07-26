# Unit scaling modes — design notes & roadmap

Status: **exploratory / not scheduled.** The `x_unit_scaling` / `y_unit_scaling`
options currently accept only `""` (off) and `"si"`. This doc records ideas for
additional modes and the reasoning behind what we did and did not ship, so we
don't re-litigate it later.

## Why it's a string, not a bool

`*_unit_scaling` is a string precisely so more modes can be added without
touching the API. The modes below split cleanly into two implementation shapes:

- **Reuse the power-of-N machinery** (min-anchor, don't-lengthen gate, per-label
  on log, out-of-range fallback) with a different marker table and/or base:
  `si` (shipped), `bytes`/`iec`, `engineering`, `short`.
- **Need their own transform** (not power-of-1000 grouping): `percent`,
  `duration`, `currency`.

The current `si` implementation lives in `uniplot/axis_labels/label_set.py`
(`_si_divisor_and_prefix`, `_si_group`, `_apply_si_prefix`, `_format_value_si`,
plus the don't-lengthen gate in `_compute_label_strings`). Adding a
table-swap mode mostly means generalizing those `_si_*` names to take a marker
table + a "marker attaches to unit vs number" flag.

## Guiding principles (learned during design)

1. **The feature is coolest on log axes.** Multi-decade log axes are exactly
   where prefixes shine (`10^-6` → `1µg`). On linear axes scaling is a mild
   convenience and the don't-lengthen gate often keeps the base unit anyway.
   Prefer modes that also help on log axes.
2. **Locale-neutral wins.** SI is the same everywhere (`km` in Berlin and
   Boston) — that's why it feels natural and is safe to ship. The more a mode
   depends on locale, the more it belongs behind an explicit option and probably
   an optional dependency.
3. **Only apply a marker if it doesn't lengthen the label** (already implemented
   for SI). Keep this for any new marker mode.
4. **Anchor a linear axis's single marker to the smallest nonzero label**, so no
   label drops below 1 in the chosen unit. Log axes format each label
   independently.

## Candidate modes

### Tier 1 — natural siblings of SI (locale-safe, reuse machinery)

- **`"bytes"` / `"iec"`** — strongest next candidate. `KiB`, `MiB`, `GiB`
  (base **1024**, IEC 80000-13), plus a base-1000 variant (`kB`, `MB`).
  Standardized, universal, squarely in uniplot's monitoring/CLI audience
  (memory, disk, network), and — like SI — impressive on log axes. Only real
  change vs SI is a base-1024 grouping instead of 1000. Marker attaches to the
  unit (`5 MiB`).
- **`"engineering"`** — SI's power-of-1000 grouping rendered as `5.2E6` /
  `1.0E-3` instead of prefix letters. The pure-numeric sibling for when there is
  no unit or the letters aren't wanted. Reuses the exact group logic.
- **`"short"`** — `k`, `M`, `B`, `T` (short scale), up only. Real niche:
  dimensionless large **counts**, where SI's `G` reads wrong (`5G requests` looks
  like 5th-gen cellular; `5B requests` is right). Cheap to add (it's an
  SI-like table with `B` at 10⁹ and no sub-1). **Caveat:** `B` is the
  internationally ambiguous token (short vs long scale) and localizes
  (`Mrd.`/`Mio.`), so it's an English-convenience, not locale-correct. If added,
  name and document it honestly as short-scale English.

### Tier 2 — useful, but need their own transform

- **`"percent"`** — multiply by 100 and append `%` (`0.42` → `42%`). Common for
  ratios / probabilities. Simple. Minor locale wrinkle: some locales use a space
  (`42 %`, e.g. fr_FR). Liked in discussion — a good second mode to consider.
- **`"duration"`** — seconds → `1h 30m`, `2d`, etc. Mixed-radix (60/60/24), so
  it's its own beast. Note SI already covers the sub-second side (`500ms`,
  `1µs` when the unit is `s`); a duration mode only adds value above 1s. Useful
  for latency/monitoring plots.
- **`"scientific"`** — `5.2e6`. Trivial; arguably just a formatting flag rather
  than a scaling mode.

### Tier 3 — locale rabbit hole (deliberately deferred)

- **`"currency"` / `"locale"`** — considered and dropped. Currency is not one
  format; it's locale-dependent on at least three independent axes:
  1. **Symbol placement** — prefix (`$1,200`) vs suffix-with-space (`1.200 €`).
  2. **Separators swap** — US `5,200,000.00`, Germany `5.200.000,00`, France
     `5 200 000,00`.
  3. **Localized abbreviations** — `M` vs `Mio.` (German), and long/short-scale
     ambiguity of "billion".
  Worse, **CJK and Indian locales don't group by thousands at all**: Japan/China
  use 万 (10⁴) and 億/亿 (10⁸) — `5,200,000` becomes `520万`, not `5.2M`; India
  uses lakh/crore (`₹52L`, grouped `52,00,000`). So the k/M/B/T *arithmetic*
  itself is Western — you can't fix it by swapping the marker table.

  Reference (CLDR via babel), value `-5,200,000`:

  | Locale | Currency | Compact | Standard |
  |---|---|---|---|
  | US | USD | `-$5.2M` | `-$5,200,000.00` |
  | Germany | EUR | `-5,2 Mio. €` | `-5.200.000,00 €` |
  | France | EUR | `-5,2 M €` | `-5 200 000,00 €` |
  | UK | GBP | `-£5.2M` | `-£5,200,000.00` |
  | Japan | JPY | `-￥520万` | `-￥5,200,000` |
  | China | CNY | `-¥520万` | `-¥5,200,000.00` |
  | India | INR | `-₹52L` | `-₹52,00,000.00` |

  Doing this correctly means CLDR data (`babel`), which owns the whole
  number→string step (it even does compact currency), needs an explicit
  `locale=` argument (the machine locale describes the *viewer*, not the *data*,
  and auto-detect makes the same script render differently across machines), and
  should be an **optional dependency** (`uniplot[currency]`) to preserve the
  lightweight core. It is a separate feature, not a marker-table variant. Only
  worth it if there is real demand.

  Note: units belong in a title (`"Revenue [M€]"`, `"单位：万元"`,
  `"(₹ in lakhs)"`) is how these locales handle it in practice — and it already
  works today via the free-form `title`, needing nothing from the library. The
  only genuinely useful locale addition would be locale-aware **number**
  formatting (decimal/thousands separator), which is orthogonal to units.

## Suggested order if we continue

`si` (done) → `bytes`/`iec` → `percent` → (maybe) `engineering` / `short`.
Currency/locale only behind an explicit locale + optional `babel` dep.
