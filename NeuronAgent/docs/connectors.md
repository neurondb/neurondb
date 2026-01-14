# Connector Usage Guide

This guide explains how to use the various connectors in NeuronAgent for integrating with external services.

## Slack Connector

The Slack connector allows you to read messages from channels, send messages, and list available channels.

### Configuration

```yaml
connector:
  type: slack
  token: xoxb-your-slack-bot-token
  endpoint: https://slack.com/api  # Optional, defaults to this
```

### Usage

#### Connect to Slack

```go
config := connectors.Config{
    Token: "xoxb-your-token",
}
connector, err := connectors.NewSlackConnector(config)
if err != nil {
    log.Fatal(err)
}

ctx := context.Background()
if err := connector.Connect(ctx); err != nil {
    log.Fatal(err)
}
```

#### Read Messages from Channel

```go
// Path can be channel ID (C123456) or channel name (#general)
reader, err := connector.Read(ctx, "C123456")
if err != nil {
    log.Fatal(err)
}

// Read messages as JSON
messages, _ := io.ReadAll(reader)
// messages contains JSON array of Slack messages
```

#### Send Message to Channel

```go
message := strings.NewReader("Hello from NeuronAgent!")
err := connector.Write(ctx, "C123456", message)
if err != nil {
    log.Fatal(err)
}
```

#### List Channels

```go
channels, err := connector.List(ctx, "")
if err != nil {
    log.Fatal(err)
}

// channels is []string with format: "#channel_name (C123456)"
for _, ch := range channels {
    fmt.Println(ch)
}
```

### Features

- **Pagination**: Automatically handles pagination for large channel lists
- **Channel Resolution**: Supports both channel IDs and names
- **Error Handling**: Comprehensive error handling with retry logic
- **Rate Limiting**: Respects Slack API rate limits

## GitHub Connector

The GitHub connector allows you to read files and list repository contents.

### Configuration

```yaml
connector:
  type: github
  token: ghp_your_github_token
  endpoint: https://api.github.com  # Optional
```

### Usage

#### Read File from Repository

```go
config := connectors.Config{
    Token: "ghp_your_token",
}
connector, err := connectors.NewGitHubConnector(config)
if err != nil {
    log.Fatal(err)
}

// Path format: owner/repo/path/to/file
reader, err := connector.Read(ctx, "owner/repo/src/main.go")
if err != nil {
    log.Fatal(err)
}
```

#### List Repository Contents

```go
// Path format: owner/repo[/path/to/dir]
files, err := connector.List(ctx, "owner/repo/src")
if err != nil {
    log.Fatal(err)
}

// files is []string with file paths
for _, file := range files {
    fmt.Println(file)  // e.g., "src/main.go" or "src/utils/"
}
```

### Features

- **File and Directory Support**: Distinguishes between files and directories
- **Pagination**: Handles GitHub API pagination
- **Rate Limit Handling**: Proper error handling for API rate limits

## GitLab Connector

The GitLab connector allows you to read files and list repository contents.

### Configuration

```yaml
connector:
  type: gitlab
  token: glpat-your-gitlab-token
  endpoint: https://gitlab.com/api/v4  # Optional
```

### Usage

#### Read File from Repository

```go
config := connectors.Config{
    Token: "glpat_your_token",
}
connector, err := connectors.NewGitLabConnector(config)
if err != nil {
    log.Fatal(err)
}

// Path format: project_id/path/to/file
reader, err := connector.Read(ctx, "12345/src/main.go")
if err != nil {
    log.Fatal(err)
}
```

#### List Repository Contents

```go
// Path format: project_id[/path/to/dir]
files, err := connector.List(ctx, "12345/src")
if err != nil {
    log.Fatal(err)
}

// files is []string with file paths
for _, file := range files {
    fmt.Println(file)
}
```

### Features

- **Tree Structure**: Properly handles GitLab tree structure
- **Pagination**: Supports GitLab API pagination
- **Error Handling**: Comprehensive error handling

## Error Handling

All connectors implement proper error handling:

- **Connection Errors**: Retry logic for transient network errors
- **Authentication Errors**: Clear error messages for invalid tokens
- **Rate Limiting**: Proper handling of API rate limits
- **Invalid Input**: Validation of path formats and parameters

## Best Practices

1. **Token Security**: Store tokens securely, never commit them to version control
2. **Error Handling**: Always check errors and handle them appropriately
3. **Rate Limiting**: Be aware of API rate limits and implement backoff strategies
4. **Connection Pooling**: Reuse connector instances when possible
5. **Context Timeouts**: Always use context with timeouts for long-running operations

## Troubleshooting

### Slack Connector Issues

**Error: "Slack authentication failed"**
- Verify your bot token is valid
- Check that the token has necessary scopes (channels:read, chat:write, etc.)

**Error: "channel not found"**
- Ensure channel ID is correct (starts with C, D, or G)
- For channel names, ensure the bot is a member of the channel

### GitHub Connector Issues

**Error: "GitHub API error: 403"**
- Check token permissions
- Verify repository access
- Check rate limit status

**Error: "invalid GitHub path format"**
- Use format: `owner/repo/path/to/file`
- Ensure path is URL-encoded if it contains special characters

### GitLab Connector Issues

**Error: "GitLab connection failed"**
- Verify token is valid
- Check project ID is correct
- Ensure token has necessary permissions

