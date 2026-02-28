# Kaggle Discussion Posts - Ready to Post

**Date:** January 31, 2026
**Status:** Ready for manual posting
**Posts Selected:** 5 of 10

---

## Quick Reference

| # | Title | Forum | Link to Post |
|---|-------|-------|--------------|
| 1 | 5 Feature Engineering Tricks | [Getting Started](https://www.kaggle.com/discussions/getting-started) | Copy Draft 1 below |
| 2 | Med-Gemma EDA Findings | Competition Forum | Copy Draft 2 below |
| 3 | Akkadian Translation Data | Competition Forum | Copy Draft 3 below |
| 4 | RAG Systems Guide | [General](https://www.kaggle.com/discussions/general) | Copy Draft 5 below |
| 5 | Time Series Pitfalls | [Getting Started](https://www.kaggle.com/discussions/getting-started) | Copy Draft 7 below |

---

## Post 1: 5 Feature Engineering Tricks That Won Me Bronze

**Post to:** https://www.kaggle.com/discussions/getting-started

### Title: 5 Feature Engineering Tricks That Won Me Bronze

### Body:
(Copy everything from "Hey Kagglers!" to the end of Draft 1 in discussion-drafts.md)

**Quick link:** See Draft 1 in `discussion-drafts.md` (lines 18-102)

---

## Post 2: Med-Gemma Challenge: Initial EDA Findings

**Post to:** Competition discussion forum (search "Med-Gemma" on Kaggle)

### Title: Med-Gemma Challenge: Initial EDA Findings

### Body:
(Copy Draft 2 from discussion-drafts.md, lines 112-187)

---

## Post 3: Akkadian Translation: Understanding the Data

**Post to:** https://www.kaggle.com/c/deep-past-initiative-machine-translation/discussion

### Title: Akkadian Translation: Understanding the Data

### Body:
(Copy Draft 3 from discussion-drafts.md, lines 197-261)

---

## Post 4: RAG Systems: What I Learned Building One From Scratch

**Post to:** https://www.kaggle.com/discussions/general

### Title: RAG Systems: What I Learned Building One From Scratch

### Body:
(Copy Draft 5 from discussion-drafts.md, lines 396-490)

---

## Post 5: Time Series Pitfalls: Don't Random Split Your Data!

**Post to:** https://www.kaggle.com/discussions/getting-started

### Title: Time Series Pitfalls: Don't Random Split Your Data!

### Body:
(Copy Draft 7 from discussion-drafts.md, lines 617-701)

---

## How to Post

1. Go to the forum URL listed above
2. Click "New Topic" or "Start Discussion"
3. Paste the title
4. Paste the body content (markdown formatting should be preserved)
5. Add relevant tags if prompted
6. Submit

## After Posting

- [ ] Post 1 submitted
- [ ] Post 2 submitted
- [ ] Post 3 submitted
- [ ] Post 4 submitted
- [ ] Post 5 submitted
- [ ] Check for upvotes after 24 hours
- [ ] Respond to any comments

---

## Kaggle API Re-authentication

The Kaggle API returned 401 Unauthorized. To fix:

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download the new `kaggle.json`
5. Replace `~/.kaggle/kaggle.json` with the new file
6. Run `chmod 600 ~/.kaggle/kaggle.json`
7. Test with `kaggle competitions list`

Once re-authenticated, you can push notebooks with:
```bash
cd /path/to/repo/kaggle
./manage.sh push-nb    # Push all notebooks
./manage.sh push graph-neural-networks  # Push GNN specifically
./manage.sh push eda-tutorial  # Push EDA tutorial
```
