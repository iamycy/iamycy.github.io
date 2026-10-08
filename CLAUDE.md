# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Chin-Yun Yu's personal academic site (https://iamycy.github.io), a Jekyll site built on the [Academic Pages](https://github.com/academicpages/academicpages.github.io) template (itself derived from Minimal Mistakes). It is built and served by GitHub Pages; there is no test suite or linter.

## Branches and remotes

- `deploy`: the site's real content (posts, publications, talks, CV, config). Work here.
- The `upstream` remote is the template (`academicpages/academicpages.github.io`). Sync it with `scripts/sync-upstream.sh`, which first runs `git pull` for the current branch, then merges `upstream/master` into it. For the paths listed in its `PERSONAL` array (config, about/CV pages, navigation, analytics, and the content collections), it always keeps our version, and it drops upstream's sample files there. Don't use a plain `git merge upstream/master`: it conflicts on those files and silently adds sample publications to the site. If another customised file starts conflicting, add it to `PERSONAL`. Run `bundle install` after a sync in case the `Gemfile` changed.

## Commands

```bash
bundle install                                # Ruby deps (Gemfile.lock is gitignored)
bundle exec jekyll serve -l -H localhost      # preview at localhost:4000 with live reload
bundle exec jekyll build                      # one-off build into _site/
docker compose up                             # same, in Docker (uses _config.yml,_config_docker.yml)
```

- `--future` also renders future-dated posts. `_config.yml` has `future: false`, so `_posts/2199-01-01-future-post.md` is hidden on purpose.
- `npm run build:js` regenerates `assets/js/main.min.js` from `assets/js/_main.js` and the plugins (needs `npm install`). Only needed after editing theme JS. The source JS is excluded from the Jekyll build, so only the minified file ships.
- `scripts/update_cv_json.sh` regenerates `_data/cv.json` from `_pages/cv.md`. This is only for the JSON CV variant; the nav links to the markdown CV at `/cv/`.

## Content model

Content is driven by Jekyll collections declared in `_config.yml` (`publications`, `talks`, `teaching`, `portfolio`, `music`). Per-collection layout defaults are set there too: `talks` uses the `talk` layout and everything else uses `single`. Each collection is listed by a page in `_pages/` (such as `publications.html` or `talks.html`), and the header menu is `_data/navigation.yml`.

- **Publications** (`_publications/YYYY-M-D-slug.md`): front matter has `collection: publications`, `category` (one of `books`, `manuscripts`, `conferences`, which map to headings via `publication_category` in `_config.yml`), `permalink`, `date`, `venue`, `paperurl` and `citation` (HTML-escaped). The abstract is the body.
- **Talks** (`_talks/`): `collection: talks`, `type`, `venue`, `date`, `location`, `permalink`. YouTube embeds are raw `<iframe>`s in the body.
- **Posts** (`_posts/YYYY-MM-DD-slug.md`): set an explicit `permalink: /posts/YYYY/MM/DD/slug/` and `tags`. Images for a post go in `images/<slug>/`. Downloadable files go in `files/`.
- **Music** is a single hand-written page (`_pages/music.md`), not a `_music/` directory.
- `markdown_generator/` has notebooks and scripts for generating publication and talk markdown from TSV or BibTeX. They are optional.

## Math and rendering

MathJax 4 is loaded on every page from `_includes/footer/custom.html`. Kramdown processes the markdown first, so inline math in posts is written `\\(...\\)` (double backslash) and display math as `$$ ... $$` on its own lines. Follow the existing posts' style.

The theme JS (`assets/js/_main.js`) renders fenced ` ```mermaid ` blocks as diagrams and ` ```plotly ` blocks (JSON with `data` and `layout`) as interactive charts. It loads each library from a CDN only on pages that use it, and redraws Plotly charts when the light/dark theme changes.

Site-wide head and analytics customisations go in `_includes/head/custom.html` and `_includes/analytics-providers/custom.html` (Google Analytics, `provider: "custom"`), not in the theme's core includes. Styles are SCSS in `_sass/`.

## Generated / ignored

`_site/`, `.sass-cache/`, `vendor/`, `.bundle/`, `node_modules/` and `Gemfile.lock` are build artifacts and gitignored. Don't edit `_site/`.
