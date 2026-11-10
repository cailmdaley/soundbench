# UI mockups

Mockups for the redesign in issue #2. The look comes from the NORAD screens in WarGames (1983), and the colour-coded controls from the Teenage Engineering OP-1. Open `index.html` in a browser; the mode tabs at the top of each page link the screens together.

Every screen is drawn from real engine output (the TypeScript engine at `c9020c3`). The head outlines follow `src/ui/render/anatomy.ts`. The pressure, pitch, spectra, gestural score, beat and creature call are simulated, not drawn by hand.

## Colours

- **Blue** is the body: outlines, grids and titles.
- **Amber** is sound: pressure, flow and spectra.
- **Green, red, pink and white** are the four controls. Each screen maps its own four controls onto these colours and draws what they move in the same colour.

## Rebuild

```sh
bun docs/mockups/gen/play.ts        # also speak, sing, beatbox, bestiary, index → docs/mockups/<name>.html
bun docs/mockups/gen/shoot.ts "$PWD/docs/mockups/play.html" /tmp/play 2   # screenshot at 2×
```

The scripts import the TypeScript engine, so they will stop working when the Rust engine replaces it. The HTML pages are self-contained and keep working.

The font is AVHershey Simplex by Stewart C. Russell (CC0 / WTFPL), from A. V. Hershey's public-domain vector fonts (1967).
