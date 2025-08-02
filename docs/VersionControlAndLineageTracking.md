# Version Control & Lineage Tracking  
Establish a rigorous version control and lineage-tracking framework...  
(Refer docs/VersionControlAndLineageTracking.md)

```markdown
# Version Control & Lineage Tracking

## 1. Purpose  
Establish a rigorous version control and lineage-tracking framework for all code, models, containers, and data artifacts in the All-in-One IoT Edge project. Ensures full traceability from source changes through builds, deployments, and published results.

---

## 2. Scope  
Applies to:
- **Source code** (Python, C/C++, MATLAB)  
- **Model artifacts** (LSTM weights, TFLite/MicroTVM bundles, autoencoder)  
- **Container images** (server, client)  
- **Datasets & configuration** (synthetic & real sensor data, hyperparameters)  
- **Documentation & diagrams**  

---

## 3. Branching Strategy  

| Branch Type      | Naming              | Purpose                                                 |
|------------------|---------------------|---------------------------------------------------------|
| **main**         | `main`              | Production-ready releases; only merges from `release/*` |
| **develop**      | `develop`           | Integration of completed features; base for new work    |
| **feature**      | `feature/<name>`    | New features or experiments; branch off `develop`       |
| **release**      | `release/vX.Y.Z`    | Stabilize a new version; bugfixes & docs updates        |
| **hotfix**       | `hotfix/vX.Y.Z+1`   | Emergency fixes to `main`; merge back into `develop`    |
| **chore**        | `chore/<area>-<desc>` | Non-functional updates (docs, CI, tooling)             |

### Workflow  
1. **Feature development** on `feature/*` → PR → review → merge into `develop`.  
2. **Release preparation**: cut `release/vX.Y.Z` → finalize tests/docs → merge into `main` & tag.  
3. **Hotfixes**: branch off `main` → patch → merge back into `main` and `develop`.  

---

## 4. Commit & Pull Request Conventions  

- **Commit Messages** follow [Conventional Commits](https://www.conventionalcommits.org/):  
```

<type>(<scope>): <short description> <BLANK LINE>
\[optional detailed description]

```
- **Types**: `feat`, `fix`, `chore`, `docs`, `perf`, `refactor`, `test`  
- **Scope**: module or folder (e.g. `client`, `matlab`, `CI`)  

- **Examples**:  
```

feat(client): add TLS support to Flower client

chore(ci): bump Python version to 3.11

````

- **Pull Requests**:
- Title follows commit convention.  
- Link to related issue (`#123`).  
- Checklist in PR description: tests, docs, changelog entry.

---

## 5. Tagging & Semantic Versioning  

- **Tags** on `main` use `v<MAJOR>.<MINOR>.<PATCH>` (SemVer).  
- **Tag annotated** with release notes and date.  
- **Pre-release tags** (alpha/beta) use `vX.Y.Z-alpha.N` or `-beta.N`.  

### Release Process  
1. Merge `release/vX.Y.Z` into `main`.  
2. Run CI full pipeline.  
3. Tag commit: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`  
4. Push tag: `git push origin vX.Y.Z`  
5. Publish Docker images and artifacts with the same tag.  

---

## 6. Artifact Lineage  

| Artifact Type    | Naming Convention                            | Stored In           |
|------------------|----------------------------------------------|---------------------|
| **Model Weights**   | `lstm-vX.Y.Z.h5`                            | `/client/models/`   |
| **TFLite Bundle**   | `ae-ttv-vX.Y.Z.tflite`                      | `/client/models/`   |
| **MicroTVM Module** | `ae-tvm-vX.Y.Z.tar.gz`                      | `/client/tvm_artifacts/` |
| **Docker Images**   | `iot-edge/server:vX.Y.Z`, `iot-edge/client:vX.Y.Z` | Docker registry      |
| **Datasets**        | `envdata-vYYYYMMDD.csv`                      | `/data/`            |

- **Model Registry**: Maintain a JSON index (`models/index.json`) that maps version tags to artifact checksums and training data hash.  
- **Container Digest**: Record image digests in `deploy/images.json` for exact reproduction.  

---

## 7. Traceability & Metadata  

- **CHANGELOG.md** at repo root, auto-generated from commit messages.  
- **Metadata files** adjacent to artifacts, e.g., `lstm-vX.Y.Z.json`:
```json
{
  "version": "v1.2.3",
  "commit": "abcdef123456",
  "date": "2025-08-02T10:15:00Z",
  "data_hash": "sha256:...",
  "parameters": { "epochs": 5, "batch_size": 32 }
}
````

* **Issue link** in metadata (`"issue": 456`) for audit.

---

## 8. Tooling & Automation

* **Git Hooks** (pre-commit): enforce commit message format and branch naming.
* **CI Tasks**:

  * Validate tag and version consistency (`package.json`, `models/index.json`, `CHANGELOG.md`).
  * Auto-bump version on merge to `main` and generate draft release notes.
* **Artifact Publishing**: scripts to push models to registry and record lineage JSON.

---

## 9. Compliance & Audit

* **Audit Logs**: GitHub audit log records merges, tag pushes, and PR approvals.
* **Retention**: Keep all tags and branches indefinitely; clean up stale feature branches monthly.
* **Access Control**: Protect `main` branch; require PR reviews and passing CI.

---

## 10. Review & Evolution

* **Quarterly Review**: Validate that branch strategy and lineage practices meet project needs.
* **Process Improvements**: Update this document and automation scripts in sync with evolving workflows.

---

*Maintained by the DevOps Engineer & Extension Steward*
*Last Updated: August 2, 2025*

```
::contentReference[oaicite:0]{index=0}
```
