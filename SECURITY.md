# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible
for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report (suspected) security vulnerabilities to
**[your.email@example.com](mailto:your.email@example.com)**. You will receive a response from
us within 48 hours. If the issue is confirmed, we will release a patch as soon
as possible depending on complexity but historically within a few days.

## Security Considerations

When using ContextForge, please consider the following security aspects:

### API Keys
- Never commit API keys to version control
- Use environment variables or secure secret management
- Rotate API keys regularly
- Use the principle of least privilege for API access

### Memory Storage
- Be cautious when storing sensitive information in memory stores
- Consider encryption for persistent storage (SQLite)
- Implement proper access controls for memory data
- Be aware of data retention policies

### Tool Execution
- Validate all tool inputs and outputs
- Implement proper sandboxing for tool execution
- Be cautious with tools that have system access
- Log tool executions for audit purposes

### Network Security
- Use HTTPS for all API communications
- Validate SSL certificates
- Implement proper timeout and retry logic
- Consider rate limiting for API calls

## Best Practices

1. **Input Validation**: Always validate user inputs before processing
2. **Output Sanitization**: Sanitize outputs before displaying to users
3. **Error Handling**: Don't expose sensitive information in error messages
4. **Logging**: Log security events but avoid logging sensitive data
5. **Dependencies**: Keep dependencies updated and monitor for vulnerabilities

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any potential similar problems
3. Prepare fixes for all releases still under maintenance
4. Release new versions as soon as possible
5. Prominently feature the problem in the release notes

## Comments on this Policy

If you have suggestions on how this process could be improved please submit a
pull request. 