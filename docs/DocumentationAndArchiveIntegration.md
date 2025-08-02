# Documentation & Archive Integration  
Framework for creating, organizing, publishing, and archiving docs...  
(Refer docs/DocumentationAndArchiveIntegration.md)

```markdown
# Documentation & Archive Integration

## 1. Purpose  
Establish a unified framework for creating, organizing, publishing, and archiving all project documentation and artifacts. Ensures that docs are discoverable, versioned alongside code, and that historical materials are retained in an accessible archive.

---

## 2. Scope  
Applies to:
- **User-facing docs** (README, Quickstarts, Tutorials)  
- **Technical reference** (Architecture, API, Code Requirements)  
- **Governance & Process** (Roadmaps, Governance docs)  
- **Release artifacts** (Datasheets, Models, Container Manifests)  
- **Archive storage** (previous versions, deprecated docs)

---

## 3. Documentation Structure  

```

/docs
├── getting-started.md      # Quickstart guides
├── architecture.md         # High-level system overview
├── code-requirements.md    # Coding standards & dependencies
├── governance              # Process & naming conventions
│   ├── folder-structure.md
│   ├── version-control.md
│   └── ...
├── tutorials               # Step-by-step how-tos
│   ├── federated-learning.md
│   ├── tinyml-deployment.md
│   └── matlab-integration.md
├── reference               # API & CLI references
│   ├── server-api.md
│   ├── client-cli.md
│   └── config-schema.md
└── images                  # Diagrams and flowcharts

````

- **Markdown-first**: All docs written in Markdown for GitHub Pages and in-repo browsing.  
- **Diagrams**: Store source files (e.g. `.drawio`, `.svg`) alongside exported PNG/SVG in `docs/images/`.  
- **Cross-references**: Use relative links (e.g. `[Architecture](architecture.md)`) to maintain integrity across branches.

---

## 4. Authoring & Review Workflow  

1. **Create or Update Doc**  
   - Branch from `develop` → `feature/docs-<topic>`.  
   - Add or edit files under `docs/`.  
   - Update `SUMMARY.md` or table-of-contents if using MkDocs.  

2. **Peer Review**  
   - Submit PR labeled `docs`.  
   - At least one technical reviewer + one docs lead must approve.  

3. **Merge & Publish**  
   - On merge to `develop`, CI builds preview site (GitHub Pages staging).  
   - On release (`main`), publish to production docs URL.

---

## 5. Documentation Site Integration  

- **Static Site Generator**: MkDocs or Sphinx  
  - Config in `mkdocs.yml` (or `docs/sphinx/conf.py`)  
- **Hosting**:  
  - GitHub Pages (branch `gh-pages`)  
  - Custom domain (e.g. `docs.iot-edge.org`)  
- **CI Pipeline**:  
  ```yaml
  # .github/workflows/docs.yml
  on: [push]
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Build Docs
          run: mkdocs build
        - name: Deploy to GitHub Pages
          uses: peaceiris/actions-gh-pages@v3
          with:
            publish_dir: ./site
````

---

## 6. Archive Integration

### 6.1 Archive Folder

Create a top-level `archive/` directory for retired documents and artifacts:

```
/archive
├── v1.0
│   ├── docs/             # Docs snapshot for v1.0
│   ├── models/           # Model artifacts (lstm-v1.0.h5, etc.)
│   └── containers/       # Docker image manifests & digests
├── v0.9
│   └── ...
└── legacy/               # Pre-0.9 or experimental archives
```

* **Versioned Snapshots**: At each release, CI copies the contents of `docs/`, `models/index.json`, and `deploy/images.json` into `archive/vX.Y/`.
* **Immutable**: Archived files are read-only to prevent accidental edits.

### 6.2 Automation

* **Archive Script** (`scripts/archive_release.sh`):

  ```bash
  #!/usr/bin/env bash
  set -e
  VERSION=$1
  mkdir -p archive/$VERSION
  cp -r docs archive/$VERSION/
  cp client/models/* archive/$VERSION/models/
  cp deploy/images.json archive/$VERSION/containers/
  git add archive/$VERSION
  git commit -m "chore(archive): snapshot release $VERSION"
  ```
* **Hook in Release Workflow**: Invoke `scripts/archive_release.sh vX.Y.Z` as part of GitHub Actions post-release job.

### 6.3 Access & Retention

* **Browse in Repo**: Developers can navigate `archive/` in GitHub to view past versions.
* **Retention Policy**:

  * Keep all minor/major release archives indefinitely.
  * Prune `legacy/` entries older than two years with maintainer approval.

---

## 7. Versioning & Lineage

Each archive entry contains metadata files:

```
archive/v1.2.0/
├── docs/
├── models/
│   └── lstm-v1.2.0.json   # {version, commit, data_hash, date}
└── containers/
    └── server-v1.2.0.json # {tag, digest, build_date}
```

* Enables precise traceability from release to source commit and artifact checksums.

---

## 8. Best Practices

* **Consistent Metadata**: Always generate `.json` metadata alongside any archived artifact.
* **Manual Review**: Before running archive script, ensure release tests & docs are finalized.
* **Documentation Version Banner**: In each archived docs set, include a banner:

  > *You are viewing archived docs for vX.Y.Z. For the latest, see \[our docs site].*

---

*Maintained by Documentation Lead & DevOps Engineer — Last updated August 2, 2025*

```
::contentReference[oaicite:0]{index=0}
```
