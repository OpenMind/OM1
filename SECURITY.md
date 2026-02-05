# Security Policy

## Supported Versions

We release security updates for the following versions of OM1:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The OpenMind team takes security bugs in OM1 seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report a Security Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@openmind.org**

Include the following information:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

When you report a security issue, you can expect:

1. **Acknowledgment**: We'll acknowledge receipt of your report within 48 hours
2. **Assessment**: We'll assess the issue and determine its severity
3. **Updates**: We'll keep you informed of our progress
4. **Fix**: We'll work on a fix and coordinate disclosure timing with you
5. **Credit**: With your permission, we'll credit you in our security advisories

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium: Within 90 days
  - Low: Next planned release

## Security Best Practices for OM1 Users

### API Keys
- **Never commit API keys** to version control
- Use `.env` files and add them to `.gitignore`
- Rotate API keys regularly
- Use different keys for development and production

### Network Security
- Use HTTPS/TLS for all API communications
- Implement proper firewall rules for robot communications
- Use VPNs when connecting to robots over public networks

### Robot Safety
- Implement physical e-stop mechanisms
- Use rate limiting for movement commands
- Validate all sensor inputs
- Implement timeout mechanisms for all operations

### Docker Security
- Keep Docker images updated
- Don't run containers as root when possible
- Use Docker secrets for sensitive data
- Scan images for vulnerabilities

## Known Security Considerations

### Zenoh/ROS2/CycloneDDS Communications
- By default, communications may not be encrypted
- For production deployments, implement proper security:
  - Enable DDS Security
  - Use Zenoh with TLS
  - Implement authentication mechanisms

### LLM API Keys
- LLM providers rate-limit and monitor for abuse
- Protect API keys as they can incur costs
- Monitor usage regularly

### WebSim Debug Interface
- WebSim (localhost:8000) is for debugging only
- Do NOT expose WebSim to public networks
- Use firewall rules to restrict access

## Security Updates

Security updates will be announced via:
- GitHub Security Advisories
- Release notes
- Email notifications (if subscribed)

## Responsible Disclosure Policy

We request that you:
- Give us reasonable time to fix issues before public disclosure
- Make a good faith effort to avoid privacy violations
- Do not access or modify user data without permission
- Do not exploit vulnerabilities beyond proof-of-concept

In return, we commit to:
- Respond promptly to your report
- Keep you informed of our progress
- Credit you for your discovery (if desired)
- Not pursue legal action against researchers who follow this policy

## Security Hall of Fame

We recognize and thank the following security researchers:

*(This section will be updated as vulnerabilities are responsibly disclosed and fixed)*

## Contact

For security-related questions or concerns:
- **Email**: security@openmind.org
- **General inquiries**: https://github.com/OpenMind/OM1/discussions

---

**Last Updated**: January 2026
```

5. **Commit message:** `docs: add security policy`
6. **Description:**
```
Added SECURITY.md outlining:

**Reporting Process:**
- How to report vulnerabilities privately
- Response timeline commitments
- Responsible disclosure guidelines

**Supported Versions:**
- Version support matrix
- Update policy

**Best Practices:**
- API key security
- Network security recommendations
- Robot safety considerations
- Docker security guidelines

**Known Considerations:**
- Zenoh/ROS2/DDS security notes
- LLM API key protection
- WebSim debug interface warnings

This establishes a professional security policy and provides clear guidelines for responsible disclosure, improving project credibility and user safety.
