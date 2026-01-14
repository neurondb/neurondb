# Contributing to NeuronDB

Thank you for your interest in contributing to NeuronDB!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/NeurondB.git` (replace `YOUR_USERNAME` with your GitHub username)
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Test your changes: `make clean && make && make installcheck`
6. Commit with clear messages: `git commit -m "Add feature X"`
7. Push and create a pull request

### Avoiding Merge Commits

To maintain a clean, linear git history, always use rebase when pulling changes:

```bash
# Configure git to always rebase on pull (recommended)
git config pull.rebase true

# Or use rebase explicitly when pulling
git pull --rebase

# Or use the workflows script
./scripts/neurondb-workflows.sh pull
```

**Never use `git pull` without `--rebase`** as it creates unnecessary merge commits like:
```
Merge branch 'main' of github.com:neurondb/neurondb
```

If you accidentally create a merge commit before pushing, you can remove it:
```bash
# Reset to before the merge (keeps your changes)
git reset --soft <commit-before-merge>

# Re-commit your changes
git commit -m "Your commit message"
```

## Code Standards

### Code Style Enforcement

All code must conform to the style guidelines enforced by automated tools. Code that doesn't pass style checks will be rejected.

#### C Code (NeuronDB Extension)

- **Style**: 100% PostgreSQL C coding standards
- **Formatting Tool**: `pgindent` (PostgreSQL's official formatter)
- **Check Tool**: `pgident` (identifier style checker)

**Formatting:**
```bash
# Format all C files
cd NeuronDB
./scripts/neurondb_format.sh

# Check formatting without modifying files
./scripts/neurondb_format.sh --check

# Show diff of formatting changes
./scripts/neurondb_format.sh --diff
```

**Style Rules:**
- Use tabs for indentation (PostgreSQL standard)
- Only C-style `/* */` comments (no `//` comments)
- Variables declared at start of function (C89/C99 compliance)
- Prefix all functions with `neurondb_`
- Include copyright and file description in headers
- Maximum line length: 80-90 characters

**CI Enforcement:**
- Formatting is checked automatically in CI
- PRs with formatting violations will fail CI checks

#### Go Code (NeuronAgent, NeuronMCP)

- **Formatting Tool**: `gofmt` (standard Go formatter)
- **Linting Tool**: `golangci-lint` (recommended)

**Formatting:**
```bash
# Format all Go files
go fmt ./...

# Or use gofmt directly
gofmt -w .
```

**Linting:**
```bash
# Install golangci-lint (if not installed)
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Run linter
golangci-lint run ./...
```

**Style Rules:**
- Follow standard Go formatting (`gofmt`)
- Use `golint` guidelines for naming and structure
- Maximum line length: 120 characters (Go standard)
- Prefer explicit error handling
- Use meaningful variable names

**CI Enforcement:**
- Formatting and linting checked automatically in CI
- PRs with violations will fail CI checks

#### TypeScript/JavaScript Code (NeuronMCP, NeuronDesktop)

- **Formatting Tool**: `prettier` (recommended) or `eslint --fix`
- **Linting Tool**: `eslint`

**Formatting:**
```bash
# Using prettier
npm run format

# Using eslint
npm run lint -- --fix
```

**Linting:**
```bash
# Run linter
npm run lint

# Fix auto-fixable issues
npm run lint -- --fix
```

**Style Rules:**
- Follow ESLint configuration
- Use 2 spaces for indentation (JavaScript/TypeScript standard)
- Maximum line length: 100 characters
- Prefer `const` over `let`, avoid `var`
- Use meaningful variable and function names

**CI Enforcement:**
- Formatting and linting checked automatically in CI
- PRs with violations will fail CI checks

### Summary Table

| Language | Formatting Tool | Linting Tool | Max Line Length | Indentation |
|----------|----------------|--------------|-----------------|-------------|
| C | `pgindent` | `pgident` | 80-90 | Tabs |
| Go | `gofmt` | `golangci-lint` | 120 | Tabs |
| TypeScript/JS | `prettier`/`eslint` | `eslint` | 100 | 2 spaces |

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

Before submitting a pull request, ensure you've completed all items in the PR checklist below.

### PR Checklist

Use this checklist when creating your pull request. All items must be completed:

#### Pre-Submission Checklist

- [ ] **Code Style**
  - [ ] C code formatted with `pgindent` (if applicable)
  - [ ] Go code formatted with `gofmt` (if applicable)
  - [ ] TypeScript/JavaScript code formatted with `prettier`/`eslint` (if applicable)
  - [ ] All linting checks pass locally
  - [ ] No style warnings or errors

- [ ] **Testing**
  - [ ] All existing tests pass (`make installcheck` for NeuronDB, `go test ./...` for Go code)
  - [ ] New tests added for new features
  - [ ] Tests updated for modified features
  - [ ] Edge cases and error conditions tested
  - [ ] Integration tests pass (if applicable)

- [ ] **Documentation**
  - [ ] Code comments added/updated for public APIs
  - [ ] User-facing documentation updated (README, docs/)
  - [ ] API documentation updated (if applicable)
  - [ ] Changelog updated (CHANGELOG.md or release notes)
  - [ ] Examples updated (if applicable)

- [ ] **Version Management**
  - [ ] Version numbers updated (if adding new features or breaking changes)
  - [ ] Version bump follows semantic versioning
  - [ ] Migration guides added for breaking changes

- [ ] **Build & CI**
  - [ ] Code compiles without warnings/errors
  - [ ] All CI checks pass (formatting, linting, tests)
  - [ ] Cross-platform compatibility verified (if applicable)

#### PR Description Template

Use this template for your PR description:

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Related Issues
Fixes #(issue number)
Related to #(issue number)

## Testing
Describe how you tested the changes

## Checklist
- [ ] Code style checks pass
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Version updated (if applicable)
- [ ] CI checks pass
```

### Review Process

1. **Automated Checks**: CI runs automatically on all PRs
   - Code formatting validation
   - Linting checks
   - Test suite execution
   - Build verification

2. **Code Review**: At least one maintainer must approve
   - Code quality and correctness
   - Adherence to coding standards
   - Test coverage
   - Documentation completeness

3. **Review SLA Expectations**:
   - **First review:** Within 48 hours of PR submission
   - **Follow-up reviews:** Within 24 hours of author response
   - **Review focus areas:**
     - Code correctness and edge cases
     - Performance implications
     - Security considerations
     - API contract compliance (for public APIs)
     - Test coverage adequacy

4. **Merge Requirements**:
   - All CI checks must pass
   - At least one approval from code owners
   - No merge conflicts
   - All checklist items completed

### Test Requirements

**Minimum Test Coverage:**
- **New features:** 80%+ code coverage for new code
- **Bug fixes:** Test case that reproduces and verifies fix
- **API changes:** Integration tests for public APIs

**Test Types Required:**
- **Unit tests:** For individual functions/components
- **Integration tests:** For cross-component interactions
- **Regression tests:** For SQL functions (in `sql/` directory)
- **Performance tests:** For performance-critical changes

**Running Tests:**
```bash
# NeuronDB extension tests
cd NeuronDB
make installcheck

# Go component tests
cd NeuronAgent
go test ./...

cd ../NeuronMCP
go test ./...

# Integration tests
./NeuronAgent/scripts/neuronagent-verify.sh
```

### Style Enforcement

**Pre-commit Hooks:**
We recommend setting up pre-commit hooks to catch style issues early:

```bash
# Install pre-commit (if not installed)
pip install pre-commit

# Set up hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

**Automated Enforcement:**
- CI automatically checks formatting and linting
- PRs with style violations will fail CI
- Fix style issues before requesting review

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Community

- Be respectful and constructive
- Follow PostgreSQL community guidelines
- Help others when possible
- Contact: support@neurondb.ai for questions

## Commit Message Guidelines

### General Rules

- The subject line must end with a period and should be concise and clear,
  typically not exceeding a single line.

- Write commit messages in paragraph form rather than as bullet points or
  lists, making sure to clearly communicate the content of the change.

- Do not reference specific file names or locations within the commit
  message text.

- Omit any mention of merge or cherry-pick actions, such as "Cherry-picked
  commit..." or "Merge commit...".

- Exclude any references to code cleanup, coding standards, or style
  violations (e.g., "C90 violation", "cleanup", or "coding standard changes").

- Leave out references to version compatibility or APIs, such as
  "PostgreSQL compatibility" or "API changes".

- Avoid including statements about the status of the codebase, such as
  whether it compiles correctly or if all errors have been resolved.

- Focus exclusively on what has changed in the commit. Do not explain
  why it was done, and do not comment on compliance with standards or
  practices.

- The body of the message should be written in clear paragraphs, providing
  a concise narrative that describes the change, breaking details into
  additional paragraphs as necessary for clarity.

- Each line contains max 80-90 characters.

- According to Module add NeuronDB:, NeuronMCP: or NeuronAgent: in first
  line of commit message.

### Example

```
NeuronDB: Improve embedding vector normalization logic.

This commit adjusts the normalization routine to
use a more numerically stable approach, addressing
issues with denormalized input data. Additional
refactoring ensures consistent vector sizing across all
embedding interfaces, providing clearer behavior for
callers and simplifying future maintenance.
```

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0 with Commons Clause restrictions and strict
commercial use prohibitions, the same license as the project.

This license strictly prohibits commercial use, creating companies based
on the code, and restricts source code use to personal, non-commercial
purposes only. Personal use of binaries is permitted. See the LICENSE
file for full terms.

