# Extension Stewardship  
Defines the process, policies, and responsibilities...  
(Refer docs/ExtensionStewardship.md)

````markdown
# Extension Stewardship

## 1. Purpose  
Defines the process, policies, and responsibilities for selecting, recommending, maintaining, and deprecating Visual Studio Code extensions across the All-in-One IoT Edge project. Ensures a consistent, secure, and performant development environment.

---

## 2. Scope  
Covers:
- VS Code **workspace-recommended** extensions (`.vscode/extensions.json`)  
- Dev-container and CI editors  
- Extension versioning, security, and compatibility  
- Onboarding and deprecation procedures  

---

## 3. Governance & Roles  

| Role               | Responsibility                                                 |
|--------------------|----------------------------------------------------------------|
| **Extension Steward**   | Curate, review, and approve extension changes                |
| **DevOps Engineer**     | Automate version pinning, monitor for updates/vulnerabilities |
| **Team Members**        | Propose new extensions; report issues or conflicts           |

- All changes to `.vscode/extensions.json` require a PR labeled `chore(extensions)` and two approvals (Steward + one maintainer).  

---

## 4. Recommended Extensions  

```jsonc
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",                     // Python support
    "ms-vscode-remote.remote-ssh",          // Remote SSH
    "ms-vscode-remote.remote-containers",   // Dev Containers
    "ms-vscode.cmake-tools",                // CMake integration
    "ms-azuretools.vscode-docker",          // Docker support
    "MathWorks.language-matlab",            // MATLAB editing/debug
    "Gimly.vscode-matlab",                  // Octave support
    "ms-vscode.cpptools",                   // C/C++ IntelliSense
    "eamodio.gitlens"                       // GitLens
  ]
}
````

* **Pin preferred versions** in `extensions.json` comments.
* **Optional** but frequently used:

  * `ms-python.vscode-pylance`
  * `ms-azure-devops.azure-pipelines`
  * `visualstudioexptteam.vscodeintellicode`

---

## 5. Lifecycle & Update Policy

1. **Proposal**

   * Open an issue “Extension Request: `<extension-id>`” describing use case and links.
2. **Review**

   * Steward verifies license (MIT/Apache only), compatibility, and overlap with existing tools.
   * Test in Dev Container for conflicts or performance regressions.
3. **Approval**

   * Merge to `develop` once two approvals obtained.
   * Add to `extensions.json` and bump version if pinning.
4. **Monitoring**

   * Monthly check for new versions via Dependabot or `code --list-extensions --show-versions`.
   * Patch increased version only after smoke testing; commit under `chore(extensions): bump X to vY`.
5. **Deprecation**

   * Mark obsolete extensions with comment `// DEPRECATED` and propose removal in next quarter.
   * Remove after a two-sprint deprecation period following governance process.

---

## 6. Security & Compliance

* **License Review**: Only MIT, Apache 2.0, or similarly permissive extensions.
* **Vulnerability Scanning**: Integrate `codeql` or similar in CI to catch malicious or outdated extensions.
* **Telemetry & Privacy**: Avoid extensions that send usage data without opt-out.

---

## 7. Onboarding & Documentation

* **Automated Prompt**: New contributors see extension recommendations when opening the workspace.
* **`docs/EXTENSIONS.md`**: Maintains rationale, usage tips, and known conflicts.
* **Workspace Walkthrough**: Include a section in onboarding guide showing how to install, update, and disable extensions.

---

## 8. Exceptional Circumstances

* **Experimental Extensions**: Prefix with `experimental.` in `extensions.json`, expire after one sprint.
* **Project-Specific Tools**: May be scoped to a single folder via workspace-folder settings, not globally recommended.

---

## 9. Review Cadence

* **Quarterly Audit**: Steward reviews entire extension list for relevance and security.
* **Sprint Spot-Check**: Quick validation during each sprint retrospective for any reported issues.

---

*Maintained by the Extension Steward. Last updated: August 2, 2025.*

```
::contentReference[oaicite:0]{index=0}
```
