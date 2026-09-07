# The Kaggle browser path sits behind one adapter

ADR-0001 put the Kaggle **CLI** behind one interface. The **browser** path was
never covered and remained in the state the CLI had been in: `kaggle_browser`
looked like the adapter, but its interface handed callers a Playwright `page`, so
every caller owned selectors, waits and clicks — and three other modules copied
its session helpers instead of importing them.

`require_playwright`, `maybe_login`, `is_authenticated`, `is_login_prompt_visible`,
`locator_count` and `first_available` existed in four modules under the same
names: `kaggle_browser`, `dataset_metadata_sync`, `campaign_execute` and
`discussion_post`. Twenty-five duplicate definitions in total. They now live once
in `shared/_browser_session.py`, with `shared/kaggle_site.py` as the interface
above it: callers ask for Kaggle *actions* and get a `SiteOutcome`.

Playwright is imported lazily inside the adapter, the same guard `kaggle_client`
uses for the Kaggle SDK — importing either module must not require a browser.

## Considered and rejected

- **Leave `kaggle_browser` as the shared module.** It is a fine session layer and
  is kept as one. What it could not be is the seam: an interface that yields a
  `page` cannot be faked, so the UI path had no test that did not stub Playwright
  itself.
- **Fold `dataset_metadata_sync`'s field editing in too.** Its edit flow is a
  stateful wizard — `find_editor_url` → `open_section_editor` → `active_form_scope`
  → `fill_field` → `save_section`, where `save_section(apply=False)` cancels
  rather than saves, so plan and apply share one traversal. That is a genuinely
  different interface shape from the seven one-shot actions and needs its own
  design pass. Its duplicated *session* helpers are folded in here; its editing
  is not.

## Consequences

- `PlaywrightSite` takes `effects`, so a gated run performs no browser action and
  records the intent instead. Note this closes no existing hole: every script's
  own `--dry-run` already returns before the browser opens, verified per script.
  The gain is one implementation rather than five spellings, and a gate a *new*
  caller inherits rather than has to remember.
- `base_url` is injectable. With the URL hardcoded, every action that builds its
  own address was untestable without reaching kaggle.com — a real browser found
  that within a minute of the fixture test existing.
- Tests come in two layers: `FakeSite` for callers, and `PlaywrightSite` driven
  against static local HTML for the selectors. Nothing reaches kaggle.com;
  upvoting and following are irreversible and public, and the account is the
  operator's.
