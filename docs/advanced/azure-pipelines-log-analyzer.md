# Microbots Log Analyzer for Azure Pipelines

A follow-along guide for adding Microbots' `LogAnalysisBot` to an Azure Pipeline as a custom task. By the end, failing builds will get an automatic root-cause analysis printed straight into the Pipeline.

> **Who is this for?** Anyone running an Azure DevOps pipeline who wants automated analysis of build or runtime logs. The task authenticates to Azure OpenAI through an Azure Resource Manager Service Connection, creates an isolated Python venv on the agent, installs `microbots[azure_ad]`, and reports the result.

---

## Prerequisites

- Azure DevOps organization where you can install custom extensions.
- Azure Resource Manager Service Connection with permission to request tokens for the Azure OpenAI resource. See [Azure managed identity setup](azure-managed-identity-setup.md) for service connection and Azure OpenAI RBAC setup. The pipeline must be authorized to use this service connection.
- Azure Pipelines agent with `azure-cli`, `python3`, `pip` and `python3 -m venv` support. Microbots uses Docker sandboxing by default, so the agent also needs a reachable Docker-compatible daemon.
- Azure OpenAI deployment that works with Microbots and is reachable by the service connection.

---

## Step 1 — Install the Marketplace Extension in Azure DevOps (Pipelines)

Install the published Marketplace extension into the Azure DevOps organization that owns your pipelines:

1. Open the Marketplace listing:
   [Microbots Log Analyzer](https://marketplace.visualstudio.com/items?itemName=Microbots-log-analyzer.microbots-log-analyzer).
   You may be required to sign in to Marketplace to view the extension.
2. Select **Get it free**.
3. Choose the Azure DevOps organization where your pipelines run.
4. Confirm the extension appears in the organization extension page:
   `https://dev.azure.com/<organization>/_settings/extensions`. If it appears under the **Shared** section you need to install it to your org (requires Azure DevOps organization administrator access).

---

## Step 2 — Add the Task to a Pipeline YAML

See the complete sample pipeline at [microbots-log-analyzer.yml](https://github.com/microsoft/microbots/blob/main/docs/examples/azure-pipelines/microbots-log-analyzer.yml).

```yaml
- task: MicrobotsLogAnalyzer@0
  displayName: Analyze build log
  inputs:
    azureSubscription: my-azure-service-connection
    deploymentName: my-azure-openai-deployment
    endpoint: https://my-azure-openai-resource.openai.azure.com/
    apiVersion: 2025-03-01-preview
    codebasePath: $(Build.SourcesDirectory)
    logFilePath: logs/build.log
    outputFilePath: $(Build.ArtifactStagingDirectory)/microbots-log-analysis.md
    debuggingLogFile: $(Build.ArtifactStagingDirectory)/microbots-log-analysis-trajectory.log
    additionalContext: |
      This build usually fails when package version conflicts occur.
      Please consider it while analyzing the log.
    timeoutSeconds: 600
    maxIterations: 20
```

The log file must exist before `MicrobotsLogAnalyzer@0` runs. Relative `logFilePath` values are resolved from `codebasePath`; absolute paths are also supported.

`outputFilePath` is optional. When it is provided, it must be an absolute path ending in `.txt`, `.md`, or `.log`. The file does not need to exist; the task creates missing directories and replaces any existing file content with the latest LLM analysis result.

`debuggingLogFile` is optional. When it is provided, it must be an absolute path ending in `.txt`, `.md`, or `.log`. The task writes the bot's runtime trajectory to it — the step-by-step LLM thoughts, tool calls, and command outputs that show how the analysis was reached. The file does not need to exist; the task creates missing directories and overwrites any existing file content. When it is omitted, no trajectory file is written and behavior is unchanged.

`additionalContext` is optional. When provided, it is appended as extra user context for the log analysis and does not replace or override the Microbots system prompt. Maximum length: 1024 characters.

---

> ✅ **That's it!** Your pipeline will now automatically analyze the provided logs and report the root cause.

> 📑 **Reference** — the sections below are reference material for tuning and understanding the task.

## Inputs

| Input | Required | Default | Description |
|---|---:|---|---|
| `azureSubscription` | Yes | - | Azure Resource Manager service connection used for Azure CLI login. Alias: `serviceConnection`. |
| `deploymentName` | Yes | - | Azure OpenAI deployment name. |
| `endpoint` | Yes | - | Azure OpenAI endpoint, for example `https://your-resource-name.openai.azure.com/`. |
| `apiVersion` | Yes | - | Azure OpenAI API version passed to Microbots, for example `2025-03-01-preview`. |
| `codebasePath` | Yes | - | Repository or source folder Microbots can inspect while analyzing the log. |
| `logFilePath` | Yes | - | Log file path. Use an absolute path, or a relative path resolved from `codebasePath`. |
| `outputFilePath` | No | - | Absolute `.txt`, `.md`, or `.log` path where the LLM analysis result is written. Missing directories are created, and existing file contents are replaced. |
| `debuggingLogFile` | No | - | Absolute `.txt`, `.md`, or `.log` path where the bot's runtime trajectory (step-by-step LLM thoughts, tool calls, and command outputs) is written. Missing directories are created, and existing file contents are replaced. Omit to skip. |
| `additionalContext` | No | - | Additional user context appended to the log analysis prompt. Maximum length: 1024 characters. |
| `timeoutSeconds` | No | `600` | Maximum time for `LogAnalysisBot.run()`. |
| `maxIterations` | No | `20` | Maximum number of Microbots iterations. Leave unset to use the default from `LogAnalysisBot.run()`. |

---

## How It Works

1. Azure Pipelines runs the task with the `Node20_1` task handler.
2. The task logs in with the supplied Azure Resource Manager Service Connection.
3. The task creates or reuses a virtual environment (`microbots-log-analyzer-venv`).
4. The task installs `microbots[azure_ad]` into that virtual environment.
5. A short Python runner creates `LogAnalysisBot` with `AzureCliCredential`, mounts `codebasePath` as context, passes `logFilePath`, optional `additionalContext`, optional `maxIterations`, and `timeoutSeconds` to `LogAnalysisBot.run()`, and prints the analysis result.
6. If `outputFilePath` is provided, the task writes the LLM analysis result to that file, replacing any existing contents.
7. If `debuggingLogFile` is provided, the task captures the bot's runtime logs (LLM thoughts, tool calls, and command outputs) to that file, replacing any existing contents.

The task clears the Azure CLI account at the end of the run. Its task manifest also uses Azure Pipelines command restrictions so analyzed log content cannot run arbitrary logging commands or set pipeline variables.

---

## Next Steps

- **Set up authentication** → [Azure Managed Identity & Service Connection Setup](azure-managed-identity-setup.md)
- **The bot behind the task** → [LogAnalysisBot API reference](../api-reference/microbots/bot/LogAnalysisBot.md)
