# Contributing to NeuronDB

Thank you for your interest in contributing to NeuronDB!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/NeurondB.git` (replace `YOUR_USERNAME` with your GitHub username)
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Test your changes: `make clean && make && make installcheck`
6. Commit with clear messages (see [Commit Message Guidelines](#commit-message-guidelines))
7. Push and create a pull request

## Commit Message Guidelines

Commit messages must contain relevant information and follow these rules:

**General Rules**

- The subject line must end with a period and should be concise and clear, typically not exceeding a single line.
- Write commit messages in paragraph form rather than as bullet points or lists, making sure to clearly communicate the content of the change.
- Do not reference specific file names or locations within the commit message text.
- Omit any mention of merge or cherry-pick actions, such as "Cherry-picked commit..." or "Merge commit...".
- Exclude any references to code cleanup, coding standards, or style violations (e.g., "C90 violation", "cleanup", or "coding standard changes").
- Leave out references to version compatibility or APIs, such as "PostgreSQL compatibility" or "API changes".
- Avoid including statements about the status of the codebase, such as whether it compiles correctly or if all errors have been resolved.
- Focus exclusively on what has changed in the commit. Do not explain why it was done, and do not comment on compliance with standards or practices.
- The body of the message should be written in clear paragraphs, providing a concise narrative that describes the change, breaking details into additional paragraphs as necessary for clarity.
- Each line contains max 80–90 characters.

**Module Prefix**

Prefix the first line of the commit message with `NeuronDB:` for this repository.

**Example**

```
NeuronDB: Improve embedding vector normalization logic.

This commit adjusts the normalization routine to
use a more numerically stable approach, addressing
issues with denormalized input data. Additional
refactoring ensures consistent vector sizing across all
embedding interfaces, providing clearer behavior for
callers and simplifying future maintenance.
```

## Code Standards

### C Code
- **Style**: 100% PostgreSQL C coding standards
- **Comments**: Only C-style `/* */` comments
- **Variables**: Declared at start of function (C89/C99 compliance)
- **Formatting**: Tabs for indentation (PostgreSQL standard)
- **Naming**: Prefix all functions with `neurondb_`
- **Headers**: Include copyright and file description

### Build Requirements
- **Zero warnings**: Code must compile with `-Wall -Wextra` clean
- **Zero errors**: All compilation must succeed
- **All platforms**: Test on Linux and macOS
- **PostgreSQL versions**: Support PG 16, 17, 18

### Testing
- Add regression tests in `sql/` with expected output in `expected/`
- Add TAP tests in `t/` for integration testing
- Run `make installcheck` before submitting PR
- Document test cases clearly

## Pull Request Process

1. Update documentation if needed
2. Add/update tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Community

- Be respectful and constructive
- Follow PostgreSQL community guidelines
- Help others when possible
- Contact: support@neurondb.ai for questions

## License

By contributing, you agree that your contributions will be licensed under
the same proprietary license as the project (see LICENSE file in the root
directory).

