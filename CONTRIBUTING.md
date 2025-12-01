# Contributing to NeuronDB

Thank you for your interest in contributing to NeuronDB!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/NeurondB.git`
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Test your changes: `make clean && make && make installcheck`
6. Commit with clear messages: `git commit -m "Add feature X"`
7. Push and create a pull request

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
- **PostgreSQL versions**: Support PG 15, 16, 17, 18

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
the GNU Affero General Public License v3.0 (AGPL-3.0), the same license
as the project.

