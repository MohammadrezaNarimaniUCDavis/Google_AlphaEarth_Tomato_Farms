# LaTeX paper

## Build

From this directory, with a LaTeX distribution installed (TeX Live, MiKTeX):

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or manually:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Optional: set output directory to keep the repo clean:

```bash
mkdir -p build
latexmk -pdf -outdir=build main.tex
```

Add `paper/build/` to `.gitignore` if you use a local build folder outside the ignored patterns (root `.gitignore` already ignores common LaTeX artifacts in-repo).

## Figures

Place included graphics under `paper/figures/` or reference `../figures/` after adjusting `\\graphicspath` in `main.tex`.
