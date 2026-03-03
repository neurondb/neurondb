# Package Naming Standards

This document defines the standard naming conventions for NeuronDB packages across different formats and operating systems.

## Standard Naming Conventions

### DEB Packages (Debian/Ubuntu)

**Format:** `{package}_{version}_{architecture}.deb`

**Examples:**
```
neurondb_1.0.0_amd64.deb
neurondb_1.0.0_arm64.deb
neuronagent_1.2.3_amd64.deb
neurondesktop_2.0.0_amd64.deb
neuronmcp_1.5.0_amd64.deb
```

**Components:**
- **Package name**: lowercase, no version
- **Version**: semantic version (MAJOR.MINOR.PATCH)
- **Architecture**: `amd64`, `arm64`, `i386`, `all`
- **Separator**: underscore (`_`)

**Pre-release versions:**
```
neurondb_1.0.0~beta1_amd64.deb
neurondb_1.0.0~rc1_amd64.deb
neurondb_1.0.0~alpha2_amd64.deb
```

**Epoch (for version conflicts):**
```
neurondb_2:1.0.0_amd64.deb
```

### RPM Packages (RHEL/CentOS/Rocky/Fedora)

**Format:** `{package}-{version}-{release}.{dist}.{architecture}.rpm`

**Examples:**
```
neurondb-1.0.0-1.el9.x86_64.rpm
neurondb-1.0.0-1.fc39.x86_64.rpm
neuronagent-1.2.3-1.el8.x86_64.rpm
neurondesktop-2.0.0-1.el9.x86_64.rpm
neuronmcp-1.5.0-1.el9.x86_64.rpm
```

**Components:**
- **Package name**: lowercase
- **Version**: semantic version (MAJOR.MINOR.PATCH)
- **Release**: build number (starts at 1)
- **Dist tag**: `el7`, `el8`, `el9`, `fc38`, `fc39` (auto-expanded by `%{?dist}`)
- **Architecture**: `x86_64`, `aarch64`, `i686`, `noarch`
- **Separator**: hyphen (`-`)

**Pre-release versions:**
```
neurondb-1.0.0-0.1.beta1.el9.x86_64.rpm
neurondb-1.0.0-0.2.rc1.el9.x86_64.rpm
```

### macOS Packages

**PKG Format:** `{Package}-{version}-{architecture}.pkg`

**Examples:**
```
NeuronDB-1.0.0-x86_64.pkg
NeuronDB-1.0.0-arm64.pkg
NeuronAgent-1.2.3-universal.pkg
```

**DMG Format:** `{Package}-{version}.dmg`

**Examples:**
```
NeuronDB-1.0.0.dmg
NeuronAgent-1.2.3.dmg
```

### Windows Packages

**MSI Format:** `{Package}-{version}-{architecture}.msi`

**Examples:**
```
NeuronDB-1.0.0-x64.msi
NeuronDB-1.0.0-x86.msi
NeuronAgent-1.2.3-x64.msi
```

**EXE Installer:** `{Package}-{version}-setup.exe`

**Examples:**
```
NeuronDB-1.0.0-setup.exe
NeuronAgent-1.2.3-setup.exe
```

## Component-Specific Naming

### NeuronDB (PostgreSQL Extension)

**DEB:**
```
neurondb_{VERSION}_{ARCH}.deb
neurondb-pg16_{VERSION}_{ARCH}.deb    # PostgreSQL version specific
neurondb-pg17_{VERSION}_{ARCH}.deb
```

**RPM:**
```
neurondb-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
neurondb-pg16-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
neurondb-pg17-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
```

**Examples:**
```
# Generic (works with any PostgreSQL version)
neurondb_1.0.0_amd64.deb
neurondb-1.0.0-1.el9.x86_64.rpm

# PostgreSQL version specific
neurondb-pg17_1.0.0_amd64.deb
neurondb-pg17-1.0.0-1.el9.x86_64.rpm
```

### NeuronAgent (Go Application)

**DEB:**
```
neuronagent_{VERSION}_{ARCH}.deb
```

**RPM:**
```
neuronagent-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
```

**Examples:**
```
neuronagent_1.2.3_amd64.deb
neuronagent-1.2.3-1.el9.x86_64.rpm
```

### NeuronDesktop (Full-stack Application)

**DEB:**
```
neurondesktop_{VERSION}_{ARCH}.deb
```

**RPM:**
```
neurondesktop-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
```

**Examples:**
```
neurondesktop_2.0.0_amd64.deb
neurondesktop-2.0.0-1.el9.x86_64.rpm
```

### NeuronMCP (MCP Server)

**DEB:**
```
neuronmcp_{VERSION}_{ARCH}.deb
neuronmcp-go_{VERSION}_{ARCH}.deb        # Go implementation
neuronmcp-typescript_{VERSION}_{ARCH}.deb # TypeScript implementation
```

**RPM:**
```
neuronmcp-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
neuronmcp-go-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
neuronmcp-typescript-{VERSION}-{RELEASE}.{DIST}.{ARCH}.rpm
```

**Examples:**
```
neuronmcp_1.5.0_amd64.deb
neuronmcp-1.5.0-1.el9.x86_64.rpm
```

## Architecture Names

### DEB (Debian naming)
- `amd64` - 64-bit x86 (Intel/AMD)
- `arm64` - 64-bit ARM (Apple Silicon, AWS Graviton)
- `i386` - 32-bit x86
- `armhf` - ARM hard-float (Raspberry Pi)
- `all` - Architecture-independent

### RPM (Red Hat naming)
- `x86_64` - 64-bit x86 (Intel/AMD)
- `aarch64` - 64-bit ARM
- `i686` - 32-bit x86
- `armv7hl` - ARM hard-float
- `noarch` - Architecture-independent

## Distribution Tags (RPM)

### RHEL/CentOS/Rocky/AlmaLinux
- `.el7` - RHEL/CentOS 7
- `.el8` - RHEL/Rocky/AlmaLinux 8
- `.el9` - RHEL/Rocky/AlmaLinux 9

### Fedora
- `.fc38` - Fedora 38
- `.fc39` - Fedora 39
- `.fc40` - Fedora 40

### OpenSUSE
- `.suse` - openSUSE Leap
- `.tw` - openSUSE Tumbleweed

## Version Numbering

### Semantic Versioning
Format: `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`

**Examples:**
- `1.0.0` - Stable release
- `1.0.0-beta1` - Beta release
- `1.0.0-rc1` - Release candidate
- `1.0.0-alpha2` - Alpha release
- `1.0.0+git.abc123` - Build with Git commit

### Version Ordering (DEB)
Uses tilde (`~`) for pre-release versions to sort before release:
```
1.0.0~alpha1 < 1.0.0~beta1 < 1.0.0~rc1 < 1.0.0
```

### Version Ordering (RPM)
Uses `0.X` release number for pre-release versions:
```
1.0.0-0.1.alpha1  (pre-release)
1.0.0-0.2.beta1   (pre-release)
1.0.0-1           (stable release)
1.0.0-2           (stable release with fixes)
```

## File Size Notation

When documenting packages, use human-readable sizes:
```
neurondb_1.0.0_amd64.deb (24.5 MB)
neuronagent_1.2.3_amd64.deb (8.3 MB)
neurondb-1.0.0-1.el9.x86_64.rpm (25.1 MB)
```

## Checksums

Always provide SHA256 checksums:
```
SHA256SUMS
SHA256SUMS.asc  (GPG signed)
```

**Format:**
```
a1b2c3d4... neurondb_1.0.0_amd64.deb
e5f6g7h8... neuronagent_1.2.3_amd64.deb
```

## GitHub Release Assets

When uploading to GitHub Releases, use this structure:

```
v1.0.0/
├── neurondb_1.0.0_amd64.deb
├── neurondb_1.0.0_arm64.deb
├── neurondb-1.0.0-1.el9.x86_64.rpm
├── neurondb-1.0.0-1.fc39.x86_64.rpm
├── neuronagent_1.0.0_amd64.deb
├── neuronagent-1.0.0-1.el9.x86_64.rpm
├── neurondesktop_1.0.0_amd64.deb
├── neurondesktop-1.0.0-1.el9.x86_64.rpm
├── neuronmcp_1.0.0_amd64.deb
├── neuronmcp-1.0.0-1.el9.x86_64.rpm
├── SHA256SUMS
└── SHA256SUMS.asc
```

## Installation Commands

### DEB Packages
```bash
# Single package
sudo dpkg -i neurondb_1.0.0_amd64.deb
sudo apt-get install -f  # Fix dependencies

# Using apt (from repository)
sudo apt install neurondb

# Using dpkg with dependencies
sudo apt install ./neurondb_1.0.0_amd64.deb
```

### RPM Packages
```bash
# Single package
sudo rpm -ivh neurondb-1.0.0-1.el9.x86_64.rpm

# Using dnf (RHEL 8+, Fedora)
sudo dnf install neurondb-1.0.0-1.el9.x86_64.rpm

# Using yum (RHEL 7)
sudo yum localinstall neurondb-1.0.0-1.el7.x86_64.rpm
```

## Package Repository Structure

### DEB Repository (APT)
```
deb https://packages.neurondb.com/deb/ stable main
```

### RPM Repository (YUM/DNF)
```
[neurondb]
name=NeuronDB Repository
baseurl=https://packages.neurondb.com/rpm/el$releasever/$basearch/
gpgcheck=1
enabled=1
```

## Standards Compliance

- **DEB**: Follows [Debian Policy Manual](https://www.debian.org/doc/debian-policy/)
- **RPM**: Follows [Fedora Packaging Guidelines](https://docs.fedoraproject.org/en-US/packaging-guidelines/)
- **Versioning**: Follows [Semantic Versioning 2.0.0](https://semver.org/)

## Quick Reference

| Format | Example | Use Case |
|--------|---------|----------|
| DEB | `neurondb_1.0.0_amd64.deb` | Ubuntu, Debian, Mint |
| RPM | `neurondb-1.0.0-1.el9.x86_64.rpm` | RHEL, CentOS, Rocky, Fedora |
| PKG | `NeuronDB-1.0.0-x86_64.pkg` | macOS |
| MSI | `NeuronDB-1.0.0-x64.msi` | Windows |

---

**Last Updated**: January 5, 2026
**Document Version**: 1.0

