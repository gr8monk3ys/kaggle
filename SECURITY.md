# Security Policy

## Reporting a Vulnerability
Please report security issues privately via GitHub Security Advisories
or by opening a private channel with the repository owner.

Include:
- A clear description of the issue
- Reproduction steps
- Impact assessment
- Suggested remediation (if known)

## Response Expectations
- Initial acknowledgement: within 72 hours
- Triage decision: within 7 days
- Fix timeline: depends on severity and complexity

## Credential Handling
- Store Kaggle credentials in `~/.kaggle/kaggle.json` (outside this repository).
- Set restrictive permissions: `chmod 600 ~/.kaggle/kaggle.json`.
- Do not commit API keys, tokens, or local credential files.
- Use `kaggle.json.example` only as a placeholder template.

## Secret Exposure Response
If a secret is committed, treat it as compromised:
1. Revoke/rotate the credential immediately.
2. Remove the secret from the repository and git history.
3. Audit recent usage for abuse and update affected systems.
