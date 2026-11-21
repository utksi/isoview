# Code signing of the Windows app (SignPath)

Not active: the project has not applied to SignPath (2026-11-22). This is ready for when it does. Windows releases of
the desktop app would then be signed through SignPath's free program for open-source projects, so that Windows
(Smart App Control, SmartScreen) lets them run; the README would also need SignPath's "Code signing policy" section.
`.github/workflows/desktop.yml` signs only builds of
`app-v*` tags, in two signing requests: first `isoview.exe`, then the `.msi` and the setup `.exe` made around it.

Set-up, once SignPath Foundation has accepted the project:

1. In SignPath, create two artifact configurations with these slugs and contents:
   `exe` from `artifact-configurations/exe.xml`, `installers` from `artifact-configurations/installers.xml`.
2. Create an API token for a SignPath user with the submitter role.
3. In GitHub (Settings > Secrets and variables > Actions), add the secret `SIGNPATH_API_TOKEN` and the repository
   variables `SIGNPATH_ORGANIZATION_ID`, `SIGNPATH_PROJECT_SLUG` and `SIGNPATH_SIGNING_POLICY_SLUG` (usually
   `release-signing`). Signing stays off until `SIGNPATH_ORGANIZATION_ID` is set.
4. Release: push a tag `app-vX.Y.Z` (matching `version` in `src-tauri/tauri.conf.json`), approve both signing
   requests in SignPath, then publish the draft release the workflow makes.
