# marcdumon.github.io

Personal blog, built with [Quarto](https://quarto.org) and served by GitHub Pages at <https://marcdumon.github.io>.

## Setup

```bash
uv tool install quarto-cli    # built with Quarto 1.10.19
```

## Commands

```bash
quarto preview                # local preview with live reload
quarto render                 # build the site into _site/
quarto publish gh-pages       # build and push _site/ to the gh-pages branch
```

## Layout

- `_quarto.yml` — site config: title, navbar, theme
- `index.qmd` — home page with the post list
- `about.qmd` — about page
- `posts/<date>-<slug>/` — one folder per post: `index.qmd` or `index.ipynb`, plus its images
- `posts/_metadata.yml` — defaults for all posts (author, freeze)

## New post

1. Do the analysis in its own uv project. Save the notebook with its outputs.
2. Copy only the finished `.ipynb` and its figures to `posts/<date>-<slug>/`. Never copy secrets or raw data.
3. Put the front matter (title, date, description, categories, image) in a raw cell at the top of the notebook.
4. Quarto does not run `.ipynb` files. It shows the saved outputs.
5. For plotly charts, set `pio.renderers.default = 'plotly_mimetype+notebook_connected'` before you save.
