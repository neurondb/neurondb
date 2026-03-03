# Release Process

This document describes how NeuronDB releases are created and published.

## Release Types

- **Major** (vX.0.0): Breaking changes, major new features
- **Minor** (vX.Y.0): New features, backward-compatible
- **Patch** (vX.Y.Z): Bug fixes, security patches

## Release Checklist

### Pre-Release

- [ ] All tests passing (CI green)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with release notes
- [ ] Version numbers updated in code
- [ ] Security review completed
- [ ] Performance benchmarks run

### Creating a Release

1. **Create Release Branch**

   ```bash
   git checkout -b release/vX.Y.Z
   git push origin release/vX.Y.Z
   ```

2. **Update Version Numbers**

   - Update `VERSION` in relevant files
   - Update package versions
   - Update Docker image tags

3. **Update CHANGELOG.md**

   - Move "Unreleased" section to new version
   - Add release date
   - Include all changes since last release

4. **Create Git Tag**

   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. **CI/CD Automatically:**

   - Builds and tests on all platforms
   - Creates Docker images and pushes to GHCR
   - Builds DEB/RPM packages
   - Creates GitHub Release with artifacts

6. **Manual Steps (if needed):**

   - Verify GHCR images are published
   - Verify packages are attached to GitHub Release
   - Update documentation site
   - Announce release (blog, social media, etc.)

## Release Artifacts

Each release includes:

### Container Images (GHCR)

- `ghcr.io/neurondb/neurondb-postgres:vX.Y.Z-pg{16|17|18}-{cpu|cuda|rocm|metal}`
- `ghcr.io/neurondb/neuronagent:vX.Y.Z`
- `ghcr.io/neurondb/neurondb-mcp:vX.Y.Z`
- `ghcr.io/neurondb/neurondesktop-api:vX.Y.Z`
- `ghcr.io/neurondb/neurondesktop-frontend:vX.Y.Z`

### Packages

- DEB packages: `neurondb_X.Y.Z_amd64.deb`, `neuronagent_X.Y.Z_amd64.deb`, `neuronmcp_X.Y.Z_amd64.deb`
- RPM packages: `neurondb-X.Y.Z-1.x86_64.rpm`, etc.
- Checksums: `SHA256SUMS` file for all artifacts

### Documentation

- Release notes (in GitHub Release and CHANGELOG.md)
- Migration guides (if applicable)
- Upgrade instructions

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward-compatible)
- **PATCH**: Backward-compatible bug fixes

## Pre-Release Versions

- **Alpha**: `vX.Y.Z-alpha.N` (internal testing)
- **Beta**: `vX.Y.Z-beta.N` (public testing)
- **RC**: `vX.Y.Z-rc.N` (release candidate)

## Release Schedule

- **Major releases**: As needed (breaking changes)
- **Minor releases**: Quarterly (new features)
- **Patch releases**: As needed (bug fixes, security)

## Rollback Procedure

If a release has critical issues:

1. **Immediate**: Mark GitHub Release as "pre-release" or add warning
2. **Short-term**: Create patch release with fixes
3. **Long-term**: Update documentation with known issues

## Release Communication

- **GitHub Release**: Detailed release notes
- **CHANGELOG.md**: Full changelog entry
- **Documentation**: Update installation guides with new versions
- **Announcements**: Blog post, social media (optional)

## Emergency Releases

For critical security fixes:

1. Create patch release immediately
2. Notify security mailing list
3. Update SECURITY.md if needed
4. Fast-track review process

## Related Documentation

- [CHANGELOG.md](CHANGELOG.md) - Full changelog
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [SECURITY.md](SECURITY.md) - Security policy
- [Container Images](docs/deployment/container-images.md) - Image documentation

