# Brand tokens

> I couldn't read your Canva file (the edit URL requires login → HTTP 403). Replace every
> `<<FILL FROM CANVA>>` below with your real values. To let me pull them for you, either paste
> the hex/font names here, or export the Canva as PNG/PDF and drop it in this folder.

Until then the design uses a **five-flavour (五味 / 五行) placeholder palette** — genuinely
on-brand for a 計五味 app, and easy to swap.

## Logo

- Wordmark: `<<FILL FROM CANVA — 計五味 lockup>>`
- App icon: `<<FILL FROM CANVA>>`
- Clear-space / min-size: `<<FILL FROM CANVA>>`

## Colour

### Core (placeholder — swap for Canva)
| Token | Hex | Use |
|-------|-----|-----|
| `--paper` | `#F6F1E7` | app background — warm almanac paper |
| `--paper-raised` | `#FCF9F2` | cards / raised surfaces |
| `--ink` | `#26221D` | primary text (warm near-black) |
| `--ink-soft` | `#7A7166` | secondary text |
| `--line` | `#E4DBCB` | hairlines, borders |
| `--seal` | `#C0453A` | cinnabar 朱砂 — primary accent, used sparingly (like a stamp) |

### Five-flavour accent system (五味 → 五行 → 五色)
Each tongue finding is tagged to the flavour/element it's traditionally associated with.
This *is* the brand system — not decoration.

| Flavour | Element | Token | Hex |
|---------|---------|-------|-----|
| 酸 sour | 木 wood | `--wood` | `#5E7F6E` (celadon green) |
| 苦 bitter | 火 fire | `--fire` | `#C0453A` (cinnabar) |
| 甘 sweet | 土 earth | `--earth` | `#D6A24E` (amber) |
| 辛 pungent | 金 metal | `--metal` | `#B9B2A6` (warm grey) |
| 鹹 salty | 水 water | `--water` | `#3C4A57` (slate blue-black) |

### Semantic (NOT the accent — for state only)
| Token | Hex | Meaning |
|-------|-----|---------|
| `--ok` | `#5E7F6E` | within typical range |
| `--note` | `#D6A24E` | worth noticing |
| `--attn` | `#C0453A` | most distinctive sign |

### Dark theme (warm, not inverted)
| Token | Hex |
|-------|-----|
| `--paper` | `#1E1B17` |
| `--paper-raised` | `#262119` |
| `--ink` | `#EDE6D8` |
| `--ink-soft` | `#A79E8F` |
| `--line` | `#38322A` |

## Type

CJK web-fonts can't be inlined (too large), so the mockup uses a system stack. Swap for the
Canva faces if they're licensed for app use.

| Role | Placeholder stack | Notes |
|------|-------------------|-------|
| Display / headings | `"Noto Serif TC", "Songti TC", "STSong", serif` | almanac / 通勝 voice |
| Body / UI | `"PingFang HK", "Noto Sans TC", system-ui, sans-serif` | gentle, readable |
| Data / numerals | body stack + `font-variant-numeric: tabular-nums` | scores, dates |

- Canva display face: `<<FILL FROM CANVA>>`
- Canva body face: `<<FILL FROM CANVA>>`

## Voice

Gentle, pressure-free, **Cantonese**. "睇下今日條脷" not "Begin diagnostic scan." Never
clinical/alarming. Every reading framed as **traditionally associated with…** (educational,
non-diagnostic).
