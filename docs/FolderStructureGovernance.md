# Folder Structure Governance  
Rules and processes for directory hierarchy organization...  
(Refer docs/FolderStructureGovernance.md)

```markdown
# Folder Structure Governance

## 1. Purpose  
Establish clear rules and processes for creating, naming, organizing, and maintaining the directory hierarchy across all repositories in the IoT Edge project. Ensures consistency, discoverability, and long-term maintainability.

---

## 2. Guiding Principles  
1. **Clarity & Purpose**  
   - Every folder’s name must clearly reflect its contents and role.  
   - Avoid ambiguous or overly generic names (e.g. prefer `client/sensing` over `stuff`).  
2. **Modularity & Separation of Concerns**  
   - Group code and assets by functional area (e.g. FL server vs. client vs. docs).  
3. **Scalability**  
   - Structure must accommodate new sub-projects (e.g. new sensor drivers, protocols).  
4. **Governance & Accountability**  
   - Changes to the top-level structure require review and approval by maintainers.  

---

## 3. Top-Level Directory Taxonomy  

```

/
├── server/       ← Federated-learning aggregator
├── client/       ← Edge-node code: sensing, ML, TinyML, MATLAB calls
├── matlab/       ← MATLAB scripts, Live Scripts, Simulink models, codegen outputs
├── docs/         ← Architecture docs, user guides, onboarding materials
├── scripts/      ← DevOps scripts: provisioning, data gen, utilities
├── tests/        ← Unit, integration, and performance tests
├── .github/      ← GitHub workflows, issue & PR templates, funding files
├── .devcontainer/← VS Code devcontainer config
├── .vscode/      ← Workspace settings, tasks.json, extensions.json, launch.json
├── README.md     ← Project overview & quickstart
├── LICENSE       ← Project license
└── CONTRIBUTING.md ← Contribution guidelines

```

---

## 4. Naming Conventions  

| Level        | Pattern                           | Example                 |
|--------------|-----------------------------------|-------------------------|
| **Top-Level**   | lowercase, hyphen-delimited        | `federated-learning/`   |
| **Subfolder**   | lowercase, single word or noun-verb | `sensing/`, `training/` |
| **Multi-word**  | lowercase with hyphens              | `simulation-clients/`   |
| **File Extensions** | language/tool indicator if needed | `.py`, `.m`, `.slx`, `.yml` |

---

## 5. Folder Roles & Responsibilities  

| Folder         | Owner(s)                         | Contents & Conventions                                         |
|----------------|----------------------------------|----------------------------------------------------------------|
| **server/**    | FL Maintainer                    | Python FL orchestration; `requirements.txt`; `Dockerfile`     |
| **client/**    | Edge-ML Lead                     | Sensor drivers; model code; TinyML wrappers; MATLAB engine hooks |
| **matlab/**    | Controls Engineer                | `.m` scripts; `.mlx` live scripts; `.slx` models; build scripts |
| **docs/**      | Documentation Lead               | Markdown guides; architecture diagrams; onboarding tutorials   |
| **scripts/**   | DevOps Engineer                  | Provisioning, CI helper scripts; data generation utilities    |
| **tests/**     | QA Lead                          | `pytest` modules; integration test harness; performance tests |
| **.github/**   | DevOps Engineer                  | CI workflows; issue/PR templates; security policies           |
| **.devcontainer/** | DevOps Engineer             | Dockerfile & `devcontainer.json` for VS Code environment      |
| **.vscode/**   | All Contributors                 | Shared VS Code settings, tasks, and launch configs             |

---

## 6. Change Management Process  
1. **Proposal**  
   - Any structural change begins with an **Issue** in GitHub labeled `architecture`.  
   - Proposer documents rationale, new layout diagram, and migration plan.  
2. **Review**  
   - Maintainers from each affected area must review within **5 business days**.  
   - Discussion in PR comments; update proposal as needed.  
3. **Approval**  
   - At least **two approvals** required: one from primary owner, one from an adjacent domain.  
4. **Implementation**  
   - Create feature branch `chore/fs-<short-desc>`.  
   - Move or rename folders/files, update references (import paths, CI workflows, docs).  
   - Run full CI suite; ensure no regressions.  
5. **Merge & Communicate**  
   - Merge into `develop`; announce change in team Slack and update `/docs/FOLDER_GOVERNANCE.md`.  
   - Tag release if breaking for downstream tooling.

---

## 7. Deprecation & Archival  
- **Deprecation**  
  - Mark folders slated for removal with a `DEPRECATED_` prefix.  
  - Deprecation period of **two sprints** during which both old and new structures coexist.  
- **Archival**  
  - After deprecation window, remove deprecated folders in a dedicated release.  
  - Archive any necessary artifacts under `/archive/<timestamp>/…` if needed for compliance.

---

## 8. Enforcement & Automation  
- **CI Validation**  
  - A GitHub Action checks for forbidden folder names, required roots, and proper casing.  
- **Pre-commit Hooks**  
  - Validate new folders adhere to naming rules before commit.  
- **Documentation Sync**  
  - A scheduled job compares actual folder layout against `docs/FOLDER_GOVERNANCE.md` and flags drift.

---

## 9. Exceptions & Extensions  
- **Experimental Folders**  
  - Prefix with `experimental/` and limit lifespan to **one sprint**.  
- **Project-Specific Additions**  
  - Allowed under `/tools/` or `/examples/` with clear README and owner attribution.  
- **Third-Party Code**  
  - Place under `/vendor/` with license headers; maintain separate `LICENSE-vendor.md`.

---

## 10. Review Cycle  
- **Quarterly Review**  
  - Folder structure governance revisited every **3 months** to adapt for project evolution.  
- **Approval Meeting**  
  - Architecture owners convene to ratify any new governance items or updates.

---

*End of Folder Structure Governance*  
```
