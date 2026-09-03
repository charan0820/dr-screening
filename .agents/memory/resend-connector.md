---
name: Resend connector
description: How the connected Resend integration behaves in this workspace.
---

Use the connected Resend application connector through the Node SDK proxy for sending email attachments. The connected credential is restricted to sending email, so read-only endpoints such as `/domains` can return a 401 while the intended `/emails` operation remains valid.

**Why:** The workspace package index did not provide the documented Python connector package, while the Node SDK installed successfully and matches the connection’s application-code setup.

**How to apply:** Keep provider credentials out of source. Pass the generated PDF as an in-memory base64 attachment to a small Node SDK bridge, and use a configured verified sender address.