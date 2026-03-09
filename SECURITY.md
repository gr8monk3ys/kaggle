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
## Supported Versions

Security updates are provided for the latest default branch.

## Reporting a Vulnerability

Please report vulnerabilities privately via GitHub Security Advisories or by contacting the maintainer directly.
Do not open public issues for undisclosed vulnerabilities.

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

For private disclosure, use GitHub Security Advisories: [https://github.com/gr8monk3ys/kaggle/security/advisories](https://github.com/gr8monk3ys/kaggle/security/advisories).
