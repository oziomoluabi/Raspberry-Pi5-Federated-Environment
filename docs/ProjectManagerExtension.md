# Project Manager Extension  
Use VS Code Project Manager to streamline navigation...  
(Refer docs/ProjectManagerExtension.md)

````markdown
# Project Manager Extension

## 1. Purpose  
Leverage the VS Code **Project Manager** extension to streamline navigation between the multiple roots of the All-in-One IoT Edge workspace. By registering each sub-project (server, client, matlab, docs, scripts), developers can switch contexts in one keystroke without manually opening folders.

---

## 2. Installation  

1. Open VS Code’s Extensions view (⇧⌘X / Ctrl+Shift+X).  
2. Search for **Project Manager** by Alessandro Fragnani (`alefragnani.project-manager`).  
3. Click **Install**.  

---

## 3. Workspace Configuration  

Add the following to your **`.vscode/settings.json`** (workspace-scope):

```jsonc
{
  // Project Manager: list of named projects
  "projectManager.projects": [
    {
      "name": "IoT Edge › Server",
      "rootPath": "${workspaceFolder}/server",
      "isFolder": true
    },
    {
      "name": "IoT Edge › Client",
      "rootPath": "${workspaceFolder}/client",
      "isFolder": true
    },
    {
      "name": "IoT Edge › MATLAB",
      "rootPath": "${workspaceFolder}/matlab",
      "isFolder": true
    },
    {
      "name": "IoT Edge › Docs",
      "rootPath": "${workspaceFolder}/docs",
      "isFolder": true
    },
    {
      "name": "IoT Edge › Scripts",
      "rootPath": "${workspaceFolder}/scripts",
      "isFolder": true
    }
  ],
  // Automatically detect folders under the workspace root
  "projectManager.git.baseFolders": [
    "${workspaceFolder}"
  ],
  // Show project tree inline with workspace
  "projectManager.showProjectExplorer": true,
  // Optional: limit history of recent projects
  "projectManager.history": 10
}
````

* **`name`**: Friendly label shown in the Project Manager list.
* **`rootPath`**: Path to the folder (supports `${workspaceFolder}` variable).
* **`isFolder`**: Indicates this entry is a folder-based project.

---

## 4. Usage

* **Open Project List**:

  * Command Palette → **Project Manager: List Projects**
  * Or use the default keybinding:

    * **Windows/Linux**: `Ctrl+Alt+P`
    * **macOS**: `⌘+Alt+P`

* **Switch Projects**:

  * Select “IoT Edge › Client” (for example), and VS Code will open that folder in the current window.

* **Add New Sub-Projects**:

  * After creating a new folder in the workspace, update **`projectManager.projects`** with its name and path.

---

## 5. Keybindings (Optional)

Add to **`keybindings.json`** for one-keystroke access:

```jsonc
[
  {
    "key": "ctrl+alt+1",
    "command": "projectManager.listProjects",
    "when": "editorTextFocus"
  }
]
```

---

## 6. Maintenance & Best Practices

* **Onboarding**: Include a pointer to this document in `README.md` so new contributors can immediately install & configure Project Manager.
* **Updates**: When you rename or move a folder, update `projectManager.projects` accordingly.
* **Consistency**: Use the “IoT Edge › …” prefix for all entries to group them together in the list.
* **Cleaning Up**: If any sub-project is deprecated, remove its entry here and optionally archive the folder.

---

*By following these guidelines, developers will enjoy rapid context switching across the multi-root IoT Edge workspace, improving productivity and reducing navigation friction.*

```
::contentReference[oaicite:0]{index=0}
```
