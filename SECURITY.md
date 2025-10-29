# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

We recommend always using the latest version from the `main` branch.

## Reporting a Vulnerability

We take security issues seriously. If you discover a security vulnerability in OM1, please report it responsibly.

**Please do NOT:**
- Open a public GitHub issue
- Discuss it in public channels (Discord, Twitter, etc.)
- Share details until we've addressed the issue

**Please do:**
1. **Email security reports to:** security@openmind.org (or open a private security advisory on GitHub)

2. **Include the following information:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

3. **We will:**
   - Acknowledge receipt within 48 hours
   - Provide updates every 7 days until resolution
   - Credit you in the security advisory (if desired)

We aim to address critical security issues within 7 days of confirmation.

## Security Best Practices

When deploying OM1:

- **API Keys**: Never commit API keys to version control. Use `.env` files or secure secret management.
- **Network**: Use HTTPS/TLS for all network communications when possible.
- **Dependencies**: Keep dependencies up to date (`uv sync --upgrade`).
- **Access Control**: Limit access to robot hardware and APIs based on principle of least privilege.
- **Updates**: Regularly update to the latest version of OM1.

## Disclosure Policy

Once a security issue is resolved, we will:

1. Publish a security advisory on GitHub (if applicable)
2. Update the changelog/release notes
3. Communicate through appropriate channels (Discord, email, etc.)

Thank you for helping keep OM1 and its users safe!

