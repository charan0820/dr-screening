---
name: Streamlit layout containers
description: A layout constraint when combining custom HTML/CSS with native Streamlit widgets.
---

Native Streamlit widgets rendered after an HTML opening tag are not reliably nested inside that tag; Streamlit emits each element as its own block. Use self-contained HTML blocks for decoration, native `st.container(border=True)` for grouped widgets, or CSS selectors that target Streamlit’s generated containers.

**Why:** Cross-widget HTML wrappers rendered as empty visual bars in the preview while the widgets appeared outside them.

**How to apply:** Never open a decorative `<div>` in one `st.markdown` call and close it after native widgets. Keep each HTML block self-contained or use a native container.