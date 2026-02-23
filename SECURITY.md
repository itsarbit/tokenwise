# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in TokenWise, please report it responsibly.

**Do not open a public issue.** Instead, email **itsarbit@gmail.com** with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will acknowledge your report within 48 hours and aim to release a fix within 7 days for critical issues.

## Scope

TokenWise handles API keys for LLM providers. Security concerns include:
- API key exposure in logs, errors, or responses
- Prompt injection via the proxy server
- Unauthorized access to the proxy endpoints

## Supported Versions

| Version | Supported |
|---|---|
| 0.5.x | Yes |
| 0.4.x | Yes |
| < 0.4 | No |
