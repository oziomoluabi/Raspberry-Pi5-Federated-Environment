# Workspace Creation & Ritual Naming  
Process for creating new VS Code workspaces and naming conventions...  
(Refer docs/WorkspaceCreationAndRitualNaming.md)

```markdown
# Workspace Creation & Ritual Naming

## 1. Purpose  
Define a consistent process (“ritual”) for creating new Visual Studio Code workspaces and naming them, to ensure clarity, discoverability, and alignment across all team projects.

---

## 2. Workspace Folder Structure  
Each new project workspace should adhere to the multi-root layout:

```

<Project-Name>.code-workspace
/ server         ← Federated-learning aggregator code
/ client         ← Edge-node sensing, ML & TinyML routines
/ matlab         ← MATLAB/Simulink prototyping & codegen artifacts
/ docs           ← Technical docs, diagrams, onboarding guides
/ scripts        ← Utility scripts (provisioning, data generation)

```

> **Note:**  
> - The top-level `.code-workspace` file defines each folder root.  
> - Additional roots (e.g. `/tools`, `/examples`) may be added per project needs.

---

## 3. Workspace File Naming  
1. **Base name**: Use the project’s canonical short name in SCREAMING_SNAKE_CASE.  
2. **Suffix**: Append the workspace type or variant:  
   - `_DEV` for development  
   - `_QA` for test/staging  
   - `_CI` for continuous-integration scripts  
   - `_LIGHT` for minimal build (no MATLAB)  

**Examples:**  
- `IOT_EDGE_DEV.code-workspace`  
- `IOT_EDGE_QA.code-workspace`  
- `IOT_EDGE_LIGHT.code-workspace`

---

## 4. Ritual Steps for Creating a New Workspace  

1. **Initialize Repository**  
   - Create GitHub repo with the project name (e.g. `iot-edge`).
   - Add `LICENSE`, `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.

2. **Define Folder Roots**  
   - Create empty directories: `/server`, `/client`, `/matlab`, `/docs`, `/scripts`.

3. **Generate Workspace File**  
   - In VS Code, select **File → Save Workspace As…**  
   - Name it per conventions (`IOT_EDGE_DEV.code-workspace`).  
   - Include each folder root in the workspace configuration.

4. **Configure `.vscode/` Settings**  
   - Add `settings.json` to each root: specify interpreter path, linters, formatters.  
   - Create `extensions.json` recommending necessary VS Code extensions.

5. **Commit & Tag**  
   - Commit workspace and config files to `develop` branch.  
   - Tag as `workspace/initial` in Git for easy rollback.

6. **Share & Onboard**  
   - Publish workspace docs in `/docs/WORKSPACE_SETUP.md`.  
   - Run a short demo (“workspace walkthrough”) with the team.

---

## 5. Naming Conventions Checklist  

| Artifact         | Convention                                 | Example                         |
|------------------|--------------------------------------------|---------------------------------|
| Workspace file   | `<PROJECT>_<ENV>.code-workspace`           | `IOT_EDGE_DEV.code-workspace`   |
| Root folders     | lowercase, hyphen-delimited                | `server`, `edge-client`, `docs` |
| Devcontainer     | `devcontainer.json` in `.devcontainer/`    | N/A                             |
| Config scripts   | verb-noun.sh or .py                        | `setup-environment.sh`          |
| Branch names     | `feature/<short-description>`              | `feature/add-tls-support`       |
| Commits          | Conventional Commits (type(scope): subject) | `feat(client): add secure auth` |

---

## 6. Ritual Naming Guidelines  

1. **Project Acronym** at start of every major artifact (`IOT_EDGE_…`).  
2. **Environment Tag** next (`DEV`, `QA`, `PROD`, `CI`, `LIGHT`).  
3. **Descriptor** describing purpose (`WORKSPACE`, `DOCKER`, `TEST`, `PIPELINE`).  

**Full Example:**  
```

IOT\_EDGE\_PROD\_WORKSPACE.code-workspace
IOT\_EDGE\_CI\_PIPELINE.yml

```

---

## 7. Benefits of Ritual Naming  
- **Discoverability**: Files and folders instantly convey purpose and scope.  
- **Consistency**: Reduces onboarding friction; everyone follows the same pattern.  
- **Automation-Friendly**: Scripts can infer environment/role from file names.  
- **Versioning Clarity**: Tags like `workspace/initial` or `pipeline/v2` map cleanly in Git.

---

## 8. Maintenance & Evolution  
- **Periodic Review**: Revisit naming and folder structure every quarter.  
- **Deprecation Policy**: Mark old workspace variants with `_ARCHIVE` suffix before removal.  
- **Documentation**: Keep `/docs/WORKSPACE_GUIDELINES.md` in sync with actual practice.

---

*End of “Workspace Creation & Ritual Naming” guidelines.*  
```
