"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");
const tl = require("azure-pipelines-task-lib/task");
const { loginAzureRM } = require("azure-pipelines-tasks-azure-arm-rest/azCliUtility");

const DEFAULT_TIMEOUT_SECONDS = "600";
const MAX_USER_PROMPT_LENGTH = 1024;
const VENV_NAME = "microbots-log-analyzer-venv";
const VENV_READY_MARKER = ".microbots-venv-ready-v1";

function runCommand(command, args, env) {
  const result = spawnSync(command, args, {
    stdio: ["ignore", "inherit", "inherit"],
    env: env || process.env,
  });

  if (result.error) throw new Error(`Failed to run ${command}: ${result.error.message}`);
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} -> exit ${result.status}`);
  }
}

function input(name, required) {
  const value = tl.getInput(name, required);
  return value ? value.trim() : value;
}

function azureSubscriptionInput() {
  const value = input("azureSubscription", false) || input("serviceConnection", false);
  if (!value) throw new Error("azureSubscription is required");
  return value;
}

function resolveLogPath(codebasePath, logFilePath) {
  return path.isAbsolute(logFilePath)
    ? path.resolve(logFilePath)
    : path.resolve(codebasePath, logFilePath);
}

function getInputs() {
  const inputs = {
    serviceConnection: azureSubscriptionInput(),
    deploymentName: input("deploymentName", true),
    endpoint: input("endpoint", true),
    apiVersion: input("apiVersion", true),
    codebasePath: tl.getPathInput("codebasePath", true, true),
    logFilePath: input("logFilePath", true),
    outputFilePath: input("outputFilePath", false),
    debuggingLogFile: input("debuggingLogFile", false),
    additionalContext: input("additionalContext", false),
    timeoutSeconds: input("timeoutSeconds", false) || DEFAULT_TIMEOUT_SECONDS,
    maxIterations: input("maxIterations", false),
  };

  validateInputs(inputs);
  return inputs;
}

function validateInputs(inputs) {
  if (!fs.existsSync(inputs.codebasePath)) {
    throw new Error(`codebasePath does not exist: ${inputs.codebasePath}`);
  }

  if (!fs.statSync(inputs.codebasePath).isDirectory()) {
    throw new Error(`codebasePath must be a directory: ${inputs.codebasePath}`);
  }

  const logPath = resolveLogPath(inputs.codebasePath, inputs.logFilePath);
  if (!fs.existsSync(logPath) || !fs.statSync(logPath).isFile()) {
    throw new Error(`logFilePath does not exist: ${logPath}`);
  }
  inputs.logFilePath = logPath;

  try {
    const endpoint = new URL(inputs.endpoint);
    if (endpoint.protocol !== "https:") throw new Error();
  } catch (_) {
    throw new Error(`endpoint must be a valid HTTPS URL: ${inputs.endpoint}`);
  }

  const timeoutSeconds = Number(inputs.timeoutSeconds);
  if (!Number.isSafeInteger(timeoutSeconds) || timeoutSeconds <= 0) {
    throw new Error(`timeoutSeconds must be a positive integer: ${inputs.timeoutSeconds}`);
  }
  inputs.timeoutSeconds = String(timeoutSeconds);

  if (inputs.maxIterations) {
    const maxIterations = Number(inputs.maxIterations);
    if (!Number.isSafeInteger(maxIterations) || maxIterations <= 0) {
      throw new Error(`maxIterations must be a positive integer: ${inputs.maxIterations}`);
    }
    inputs.maxIterations = String(maxIterations);
  }

  if (inputs.outputFilePath) {
    inputs.outputFilePath = validateWritableFilePath(inputs.outputFilePath, "outputFilePath");
  }

  if (inputs.debuggingLogFile) {
    inputs.debuggingLogFile = validateWritableFilePath(inputs.debuggingLogFile, "debuggingLogFile");
  }

  if (inputs.additionalContext && inputs.additionalContext.length > MAX_USER_PROMPT_LENGTH) {
    throw new Error(`additionalContext must be ${MAX_USER_PROMPT_LENGTH} characters or fewer`);
  }
}

function validateWritableFilePath(filePath, label) {
  if (!path.isAbsolute(filePath)) {
    throw new Error(`${label} must be an absolute path: ${filePath}`);
  }

  const resolved = path.resolve(filePath);
  const extension = path.extname(resolved).toLowerCase();
  if (extension !== ".txt" && extension !== ".md" && extension !== ".log") {
    throw new Error(`${label} must end with .txt, .md, or .log: ${resolved}`);
  }

  if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
    throw new Error(`${label} must be a file path, not a directory: ${resolved}`);
  }

  return resolved;
}

async function loginWithServiceConnection(serviceConnection) {
  console.log("##[section]MicrobotsLogAnalyzer: authenticating with Azure service connection");
  const previousAzureOutput = process.env.AZURE_CORE_OUTPUT;
  const originalLoc = tl.loc;

  process.env.AZURE_CORE_OUTPUT = "none";
  tl.loc = function loc(key, ...args) {
    if (key === "LoginFailed" || key === "ErrorInSettingUpSubscription") return key;
    return originalLoc(key, ...args);
  };

  try {
    await loginAzureRM(serviceConnection);
  } catch (error) {
    throw new Error(`Azure service connection login failed for '${serviceConnection}': ${error.message || String(error)}`);
  } finally {
    tl.loc = originalLoc;
    if (previousAzureOutput === undefined) delete process.env.AZURE_CORE_OUTPUT;
    else process.env.AZURE_CORE_OUTPUT = previousAzureOutput;
  }

  console.log("##[section]MicrobotsLogAnalyzer: Azure authentication complete");
}

function venvPythonPath(venvDir) {
  return process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
}

function setupVenv() {
  const venvRoot = process.env.AGENT_TEMPDIRECTORY
    || process.env.PIPELINE_WORKSPACE
    || process.env.RUNNER_TEMP
    || "/tmp";
  const venvDir = path.join(venvRoot, VENV_NAME);
  const python = venvPythonPath(venvDir);
  const venvReadyFile = path.join(venvDir, VENV_READY_MARKER);

  if (fs.existsSync(python) && fs.existsSync(venvReadyFile)) {
    console.log(`##[section]MicrobotsLogAnalyzer: reusing Python environment at ${venvDir}`);
    return python;
  }

  if (fs.existsSync(venvDir)) fs.rmSync(venvDir, { recursive: true, force: true });

  console.log(`##[section]MicrobotsLogAnalyzer: creating Python environment at ${venvDir}`);
  runCommand("python3", ["-m", "venv", venvDir]);
  console.log("Installing Python dependencies (microbots, Azure identity)...");
  runCommand(python, ["-m", "pip", "install", "--quiet", "--upgrade", "pip"]);
  runCommand(python, ["-m", "pip", "install", "--quiet", "microbots[azure_ad]"]);
  fs.writeFileSync(venvReadyFile, new Date().toISOString());

  return python;
}

function microbotsEnvironment(inputs) {
  return Object.assign({}, process.env, {
    AZURE_OPENAI_DEPLOYMENT_NAME: inputs.deploymentName,
    AZURE_OPENAI_ENDPOINT: inputs.endpoint,
    AZURE_OPENAI_API_VERSION: inputs.apiVersion,
  });
}

function ensureOutputParentDirectory(outputFilePath, description) {
  if (!outputFilePath) return;

  const outputDirectory = path.dirname(outputFilePath);
  if (fs.existsSync(outputDirectory) && !fs.statSync(outputDirectory).isDirectory()) {
    throw new Error(`${description} parent must be a directory: ${outputDirectory}`);
  }

  fs.mkdirSync(outputDirectory, { recursive: true });
  console.log(`##[section]MicrobotsLogAnalyzer: ${description} will overwrite ${outputFilePath}`);
}

function runLogAnalyzer(python, inputs) {
  const scriptPath = path.join(__dirname, "log_analyzer_runner.py");
  const args = [
    scriptPath,
    "--codebase-path", inputs.codebasePath,
    "--log-file-path", inputs.logFilePath,
    "--timeout-seconds", inputs.timeoutSeconds,
  ];

  if (inputs.outputFilePath) args.push("--output-file", inputs.outputFilePath);
  if (inputs.debuggingLogFile) args.push("--debug-log-file", inputs.debuggingLogFile);
  if (inputs.additionalContext) args.push("--user-prompt", inputs.additionalContext);
  if (inputs.maxIterations) args.push("--max-iterations", inputs.maxIterations);

  ensureOutputParentDirectory(inputs.outputFilePath, "analysis output");
  ensureOutputParentDirectory(inputs.debuggingLogFile, "debugging log");
  runCommand(python, args, microbotsEnvironment(inputs));
}

async function run() {
  let clearAzAccount = false;

  try {
    const inputs = getInputs();
    clearAzAccount = true;
    await loginWithServiceConnection(inputs.serviceConnection);
    const python = setupVenv();
    runLogAnalyzer(python, inputs);
    tl.setResult(tl.TaskResult.Succeeded, "LogAnalysisBot completed");
  } catch (error) {
    tl.setResult(tl.TaskResult.Failed, error.message || String(error));
  } finally {
    if (clearAzAccount) {
      try { spawnSync("az", ["account", "clear"], { stdio: "ignore" }); } catch (_) {}
    }
  }
}

run();
