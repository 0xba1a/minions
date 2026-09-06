"use strict";

const assert = require("node:assert/strict");
const childProcess = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const taskDir = path.resolve(__dirname, "..");

function loadTask(options = {}) {
  const source = fs.readFileSync(path.join(taskDir, "index.js"), "utf8").replace(
    /run\(\);\s*$/,
    `module.exports = {
      runCommand,
      input,
      azureSubscriptionInput,
      resolveLogPath,
      getInputs,
      validateInputs,
      loginWithServiceConnection,
      venvPythonPath,
      ensureOutputParentDirectory,
      setupVenv,
      microbotsEnvironment,
      runLogAnalyzer,
      run,
    };`
  );

  const calls = {
    spawnSync: [],
    loginAzureRM: [],
    setResult: [],
    rmSync: [],
    writeFileSync: [],
  };
  const events = [];

  const taskLib = options.taskLib || {
    inputs: options.inputs || {},
    pathInputs: options.pathInputs || {},
    TaskResult: { Succeeded: "Succeeded", Failed: "Failed" },
    getInput(name, required) {
      const value = this.inputs[name];
      if (required && (value === undefined || value === null || value === "")) {
        throw new Error(`${name} is required`);
      }
      return value;
    },
    getPathInput(name, required) {
      const value = this.pathInputs[name] ?? this.inputs[name];
      if (required && (value === undefined || value === null || value === "")) {
        throw new Error(`${name} is required`);
      }
      return value;
    },
    loc(key, ...args) {
      return [key, ...args].join(" ");
    },
    setResult(result, message) {
      calls.setResult.push({ result, message });
      events.push({ name: "setResult", result });
    },
  };

  const mockFs = options.fs || fs;
  const mockProcess = options.process || { env: {}, platform: "linux" };
  const mockSpawnSync = options.spawnSync || ((command, args, spawnOptions) => {
    calls.spawnSync.push({ command, args, options: spawnOptions });
    events.push({ name: "spawnSync", command, args });
    return { status: 0 };
  });
  const mockLoginAzureRM = options.loginAzureRM || (async (serviceConnection) => {
    calls.loginAzureRM.push(serviceConnection);
    events.push({ name: "loginAzureRM", serviceConnection });
  });

  const module = { exports: {} };
  const context = {
    Date,
    Error,
    URL,
    __dirname: taskDir,
    console: options.console || { log() {}, warn() {}, error() {} },
    module,
    exports: module.exports,
    process: mockProcess,
    require(name) {
      if (name === "fs") return mockFs;
      if (name === "path") return path;
      if (name === "child_process") return { spawnSync: mockSpawnSync };
      if (name === "azure-pipelines-task-lib/task") return taskLib;
      if (name === "azure-pipelines-tasks-azure-arm-rest/azCliUtility") {
        return { loginAzureRM: mockLoginAzureRM };
      }
      return require(name);
    },
  };

  vm.runInNewContext(source, context, { filename: path.join(taskDir, "index.js") });
  return { task: module.exports, calls, events, taskLib, process: mockProcess };
}

function makeProjectWithLog() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "microbots-task-test-"));
  const logDir = path.join(root, "logs");
  fs.mkdirSync(logDir);
  fs.writeFileSync(path.join(logDir, "build.log"), "error log");
  return root;
}

function mockCompletedLinuxVenv(tempDir = "/tmp") {
  const venvDir = path.join(tempDir, "microbots-log-analyzer-venv");
  const python = path.join(venvDir, "bin", "python");
  const marker = path.join(venvDir, ".microbots-venv-ready-v1");
  const existing = new Set([python, marker]);
  const mockFs = {
    existsSync(filePath) { return existing.has(filePath) || fs.existsSync(filePath); },
    statSync(filePath) { return fs.statSync(filePath); },
    rmSync() {},
    readFileSync(filePath, encoding) { return fs.readFileSync(filePath, encoding); },
    mkdirSync(filePath, options) { return fs.mkdirSync(filePath, options); },
    writeFileSync(filePath, value) { return fs.writeFileSync(filePath, value); },
  };

  return { mockFs, python };
}

function writeFile(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
}

function mockMicrobotsEnvironment(options = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "microbots-runner-test-"));
  const mockModules = path.join(root, "mocks");
  const recordPath = path.join(root, "record.json");

  writeFile(path.join(mockModules, "azure", "__init__.py"), "");
  writeFile(path.join(mockModules, "azure", "identity.py"), `
class AzureCliCredential:
    pass

def get_bearer_token_provider(credential, scope):
    def token_provider():
        return "mock-token"
    token_provider.scope = scope
    token_provider.credential_type = type(credential).__name__
    return token_provider
`);
  writeFile(path.join(mockModules, "microbots.py"), `
import json
import logging
import os
from types import SimpleNamespace

class LogAnalysisBot:
    def __init__(self, model, folder_to_mount, token_provider):
        if os.environ.get("MICROBOTS_MOCK_INIT_ERROR") == "docker":
            class DockerException(Exception):
                pass
            DockerException.__module__ = "docker.errors"
            raise DockerException("Error while fetching server API version")
        self.model = model
        self.folder_to_mount = folder_to_mount
        self.token_provider = token_provider

    def run(self, **kwargs):
        logging.getLogger("microbots.MicroBot").info("MOCK_TRAJECTORY LLM tool call: ls -1 /var/log")
        record = {
            "model": self.model,
            "folder_to_mount": self.folder_to_mount,
            "token_provider_scope": getattr(self.token_provider, "scope", None),
            "token_provider_credential_type": getattr(self.token_provider, "credential_type", None),
            "token": self.token_provider(),
            "run_kwargs": kwargs,
        }
        with open(os.environ["MICROBOTS_MOCK_RECORD"], "w", encoding="utf-8") as record_file:
            json.dump(record, record_file)
        return SimpleNamespace(
            status=os.environ["MICROBOTS_MOCK_STATUS"] == "true",
            result=os.environ.get("MICROBOTS_MOCK_RESULT") or None,
            error=os.environ.get("MICROBOTS_MOCK_ERROR") or None,
        )
`);

  const env = Object.assign({}, process.env, {
    AZURE_OPENAI_DEPLOYMENT_NAME: options.deploymentName || "gpt-test",
    MICROBOTS_MOCK_RECORD: recordPath,
    MICROBOTS_MOCK_STATUS: options.status === false ? "false" : "true",
    MICROBOTS_MOCK_RESULT: Object.hasOwn(options, "result") ? options.result : "Root cause found",
    MICROBOTS_MOCK_ERROR: Object.hasOwn(options, "error") ? options.error : "",
    MICROBOTS_MOCK_INIT_ERROR: options.initError || "",
    PYTHONPATH: mockModules + (process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : ""),
  });

  return { env, recordPath };
}

async function runTaskWithMockServiceConnectionAndMockMicrobots(options = {}) {
  const codebasePath = makeProjectWithLog();
  const { mockFs, python } = mockCompletedLinuxVenv();
  const logFilePath = path.join(codebasePath, "logs", "build.log");
  const inputs = Object.assign({
    azureSubscription: "service-id",
    deploymentName: "gpt-test",
    endpoint: "https://example.openai.azure.com/",
    apiVersion: "2025-03-01-preview",
    codebasePath,
    logFilePath: "logs/build.log",
    timeoutSeconds: "600",
    maxIterations: "12",
  }, options.inputs || {});
  let runnerResult;
  let runnerRecord;
  let runnerRecordPath;
  let loaded;

  loaded = loadTask({
    fs: mockFs,
    process: {
      env: Object.assign({ AGENT_TEMPDIRECTORY: "/tmp" }, options.processEnv || {}),
      platform: "linux",
    },
    inputs,
    spawnSync(command, args, spawnOptions) {
      const commandArgs = Array.from(args || []);
      loaded.calls.spawnSync.push({ command, args, options: spawnOptions });
      loaded.events.push({ name: "spawnSync", command, args });

      if (commandArgs[0] === path.join(taskDir, "log_analyzer_runner.py")) {
        const mockMicrobots = mockMicrobotsEnvironment(Object.assign({
          deploymentName: inputs.deploymentName,
          result: "Root cause found",
        }, options.mockMicrobots || {}));
        runnerRecordPath = mockMicrobots.recordPath;
        runnerResult = childProcess.spawnSync("python", Array.from(args), {
          env: Object.assign({}, spawnOptions.env, mockMicrobots.env),
          encoding: "utf8",
        });
        runnerRecord = fs.existsSync(runnerRecordPath)
          ? JSON.parse(fs.readFileSync(runnerRecordPath, "utf8"))
          : null;
        return {
          status: runnerResult.status,
          stdout: runnerResult.stdout,
          stderr: runnerResult.stderr,
        };
      }
      return { status: 0 };
    },
  });

  await loaded.task.run();
  return { ...loaded, codebasePath, logFilePath, python, runnerResult, runnerRecord };
}

test("Input Parameter azureSubscription has serviceConnection as alias", () => {
  const primary = loadTask({ inputs: { azureSubscription: " primary " } });
  assert.equal(primary.task.azureSubscriptionInput(), "primary");

  const alias = loadTask({ inputs: { serviceConnection: " alias " } });
  assert.equal(alias.task.azureSubscriptionInput(), "alias");

  const missing = loadTask();
  assert.throws(() => missing.task.azureSubscriptionInput(), /azureSubscription is required/);
});

test("Valid Inputs Resolve Correctly: log path and numeric values", () => {
  const { task } = loadTask();
  const codebasePath = makeProjectWithLog();
  const inputs = {
    codebasePath,
    logFilePath: "logs/build.log",
    endpoint: "https://example.openai.azure.com/",
    timeoutSeconds: " 600 ",
    maxIterations: "20",
  };

  task.validateInputs(inputs);

  assert.equal(inputs.logFilePath, path.join(codebasePath, "logs", "build.log"));
  assert.equal(inputs.timeoutSeconds, "600");
  assert.equal(inputs.maxIterations, "20");
});

test("Optional Output File Path Must Be Absolute Plain Text Path", () => {
  const { task } = loadTask();
  const codebasePath = makeProjectWithLog();
  const validInputs = {
    codebasePath,
    logFilePath: "logs/build.log",
    endpoint: "https://example.openai.azure.com/",
    timeoutSeconds: "600",
  };

  assert.throws(
    () => task.validateInputs({ ...validInputs, outputFilePath: "analysis.md" }),
    /outputFilePath must be an absolute path/
  );
  assert.throws(
    () => task.validateInputs({ ...validInputs, outputFilePath: "$(Build.ArtifactStagingDirectory)/analysis.md" }),
    /outputFilePath must be an absolute path/
  );
  assert.throws(
    () => task.validateInputs({ ...validInputs, outputFilePath: path.join(codebasePath, "analysis.json") }),
    /outputFilePath must end with .txt, .md, or .log/
  );

  const inputs = { ...validInputs, outputFilePath: path.join(codebasePath, "out", "analysis.md") };
  task.validateInputs(inputs);
  assert.equal(inputs.outputFilePath, path.join(codebasePath, "out", "analysis.md"));

  const logInputs = { ...validInputs, outputFilePath: path.join(codebasePath, "out", "analysis.log") };
  task.validateInputs(logInputs);
  assert.equal(logInputs.outputFilePath, path.join(codebasePath, "out", "analysis.log"));
});

test("Optional Debugging Log File Must Be Absolute Allowed-Extension Path", () => {
  const { task } = loadTask();
  const codebasePath = makeProjectWithLog();
  const validInputs = {
    codebasePath,
    logFilePath: "logs/build.log",
    endpoint: "https://example.openai.azure.com/",
    timeoutSeconds: "600",
  };

  assert.throws(
    () => task.validateInputs({ ...validInputs, debuggingLogFile: "trajectory.log" }),
    /debuggingLogFile must be an absolute path/
  );
  assert.throws(
    () => task.validateInputs({ ...validInputs, debuggingLogFile: path.join(codebasePath, "trajectory.json") }),
    /debuggingLogFile must end with .txt, .md, or .log/
  );

  const inputs = { ...validInputs, debuggingLogFile: path.join(codebasePath, "out", "trajectory.log") };
  task.validateInputs(inputs);
  assert.equal(inputs.debuggingLogFile, path.join(codebasePath, "out", "trajectory.log"));
});

test("Invalid Inputs Are Rejected: endpoint, timeout, and maxIterations", () => {
  const { task } = loadTask();
  const codebasePath = makeProjectWithLog();
  const validInputs = {
    codebasePath,
    logFilePath: "logs/build.log",
    endpoint: "https://example.openai.azure.com/",
    timeoutSeconds: "600",
  };

  assert.throws(
    () => task.validateInputs({ ...validInputs, endpoint: "http://example.openai.azure.com/" }),
    /valid HTTPS URL/
  );
  assert.throws(
    () => task.validateInputs({ ...validInputs, timeoutSeconds: "0" }),
    /timeoutSeconds must be a positive integer/
  );
  assert.throws(
    () => task.validateInputs({ ...validInputs, maxIterations: "-1" }),
    /maxIterations must be a positive integer/
  );
  assert.throws(
    () => task.validateInputs({ ...validInputs, additionalContext: "x".repeat(1025) }),
    /additionalContext must be 1024 characters or fewer/
  );
});

test("End To End Flow Works: ServiceConnection Login and LogAnalysisBot Output is Displayed", async () => {
  const {
    calls,
    events,
    codebasePath,
    logFilePath,
    python,
    runnerResult,
    runnerRecord,
  } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    processEnv: {
        AGENT_TEMPDIRECTORY: "/tmp",
        AZURE_OPENAI_DEPLOYMENT_NAME: "stale-deployment",
        AZURE_OPENAI_ENDPOINT: "https://stale.openai.azure.com/",
        AZURE_OPENAI_API_VERSION: "stale-version",
        KEEP_ME: "yes",
    },
    mockMicrobots: { result: "The deployment returned analysis." },
  });

  assert.deepEqual(calls.setResult, [{ result: "Succeeded", message: "LogAnalysisBot completed" }]);
  const loginIndex = events.findIndex((event) => event.name === "loginAzureRM");
  const runnerIndex = events.findIndex((event) => (
    event.name === "spawnSync" && event.args[0] === path.join(taskDir, "log_analyzer_runner.py")
  ));
  assert.notEqual(loginIndex, -1);
  assert.notEqual(runnerIndex, -1);
  assert.equal(events[loginIndex].serviceConnection, "service-id");
  assert.ok(loginIndex < runnerIndex);

  const runnerCall = calls.spawnSync.find((call) => (
    call.args[0] === path.join(taskDir, "log_analyzer_runner.py")
  ));
  assert.ok(runnerCall);
  assert.equal(runnerResult.status, 0, runnerResult.stderr);
  assert.match(runnerResult.stdout, /The deployment returned analysis\./);
  assert.equal(runnerCall.command, python);
  assert.deepEqual(Array.from(runnerCall.args), [
    path.join(taskDir, "log_analyzer_runner.py"),
    "--codebase-path", codebasePath,
    "--log-file-path", logFilePath,
    "--timeout-seconds", "600",
    "--max-iterations", "12",
  ]);
  assert.equal(runnerCall.options.stdio[1], "inherit");
  assert.equal(runnerCall.options.stdio[2], "inherit");
  assert.equal(runnerCall.options.env.KEEP_ME, "yes");
  assert.equal(runnerCall.options.env.AZURE_OPENAI_DEPLOYMENT_NAME, "gpt-test");
  assert.equal(runnerCall.options.env.AZURE_OPENAI_ENDPOINT, "https://example.openai.azure.com/");
  assert.equal(runnerCall.options.env.AZURE_OPENAI_API_VERSION, "2025-03-01-preview");
  assert.equal(runnerRecord.model, "azure-openai/gpt-test");
  assert.equal(runnerRecord.token_provider_scope, "https://cognitiveservices.azure.com/.default");
  assert.deepEqual(runnerRecord.run_kwargs, {
    file_name: logFilePath,
    timeout_in_seconds: 600,
    max_iterations: 12,
  });

  const logoutCall = calls.spawnSync.at(-1);
  assert.equal(logoutCall.command, "az");
  assert.deepEqual(Array.from(logoutCall.args), ["account", "clear"]);
  assert.equal(logoutCall.options.stdio, "ignore");
});

test("Optional Additional Context Is Passed To LogAnalysisBot", async () => {
  const { calls, runnerRecord } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    inputs: { additionalContext: "Prefer dependency restore failures as the likely root cause." },
  });

  const runnerCall = calls.spawnSync.find((call) => (
    call.args[0] === path.join(taskDir, "log_analyzer_runner.py")
  ));
  assert.ok(runnerCall);
  assert.ok(Array.from(runnerCall.args).includes("--user-prompt"));
  assert.deepEqual(runnerRecord.run_kwargs, {
    file_name: path.join(runnerRecord.folder_to_mount, "logs", "build.log"),
    timeout_in_seconds: 600,
    max_iterations: 12,
    user_prompt: "Prefer dependency restore failures as the likely root cause.",
  });
});

test("Optional Debugging Log File Captures Runtime Trajectory", async () => {
  const debugRoot = fs.mkdtempSync(path.join(os.tmpdir(), "microbots-debug-test-"));
  const debuggingLogFile = path.join(debugRoot, "nested", "trajectory.log");

  const { calls, runnerResult } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    inputs: { debuggingLogFile },
  });

  const runnerCall = calls.spawnSync.find((call) => (
    call.args[0] === path.join(taskDir, "log_analyzer_runner.py")
  ));
  assert.ok(runnerCall);
  const runnerArgs = Array.from(runnerCall.args);
  assert.ok(runnerArgs.includes("--debug-log-file"));
  assert.equal(runnerArgs[runnerArgs.indexOf("--debug-log-file") + 1], debuggingLogFile);

  assert.equal(runnerResult.status, 0, runnerResult.stderr);
  assert.equal(fs.existsSync(debuggingLogFile), true);
  const trajectory = fs.readFileSync(debuggingLogFile, "utf8");
  assert.match(trajectory, /MOCK_TRAJECTORY LLM tool call: ls -1 \/var\/log/);
  assert.match(trajectory, /microbots\.MicroBot INFO/);
});

test("Debugging Log File Is Skipped When Not Provided", async () => {
  const { calls } = await runTaskWithMockServiceConnectionAndMockMicrobots();

  const runnerCall = calls.spawnSync.find((call) => (
    call.args[0] === path.join(taskDir, "log_analyzer_runner.py")
  ));
  assert.ok(runnerCall);
  assert.ok(!Array.from(runnerCall.args).includes("--debug-log-file"));
});

test("Existing Python Environment Is Reused Only After A Completed Setup", () => {
  const tempDir = path.join(path.parse(process.cwd()).root, "tmp");
  const venvDir = path.join(tempDir, "microbots-log-analyzer-venv");
  const python = path.join(venvDir, "bin", "python");
  const marker = path.join(venvDir, ".microbots-venv-ready-v1");
  const exists = new Set([
    python,
    marker,
  ]);
  const mockFs = {
    existsSync(filePath) { return exists.has(filePath); },
    rmSync() { throw new Error("rmSync should not be called"); },
    writeFileSync() { throw new Error("writeFileSync should not be called"); },
  };
  const { task, calls } = loadTask({ fs: mockFs, process: { env: { AGENT_TEMPDIRECTORY: tempDir }, platform: "linux" } });

  assert.equal(task.setupVenv(), python);
  assert.equal(calls.spawnSync.length, 0);
});

test("Incomplete Python Environment Is Deleted And Rebuilt", () => {
  const tempDir = path.join(path.parse(process.cwd()).root, "tmp");
  const venvDir = path.join(tempDir, "microbots-log-analyzer-venv");
  const python = path.join(venvDir, "bin", "python");
  const marker = path.join(venvDir, ".microbots-venv-ready-v1");
  const existing = new Set([venvDir]);
  const mockFs = {
    existsSync(filePath) { return existing.has(filePath); },
    rmSync(filePath, options) { existing.delete(filePath); calls.rmSync.push({ filePath, options }); },
    writeFileSync(filePath, value) { calls.writeFileSync.push({ filePath, value }); },
  };
  const calls = { rmSync: [], writeFileSync: [] };
  const loaded = loadTask({
    fs: mockFs,
    process: { env: { AGENT_TEMPDIRECTORY: tempDir }, platform: "linux" },
  });

  assert.equal(loaded.task.setupVenv(), python);
  assert.equal(calls.rmSync[0].filePath, venvDir);
  assert.equal(calls.rmSync[0].options.recursive, true);
  assert.equal(calls.rmSync[0].options.force, true);
  assert.deepEqual(loaded.calls.spawnSync.map((call) => [call.command, Array.from(call.args)]), [
    ["python3", ["-m", "venv", venvDir]],
    [python, ["-m", "pip", "install", "--quiet", "--upgrade", "pip"]],
    [python, ["-m", "pip", "install", "--quiet", "microbots[azure_ad]"]],
  ]);
  assert.equal(calls.writeFileSync[0].filePath, marker);
});

test("AzureRM Login Receives ServiceConnection ID", async () => {
  const { task, calls, process } = loadTask({
    process: { env: { AZURE_CORE_OUTPUT: "json" }, platform: "linux" },
  });

  await task.loginWithServiceConnection("service-id");

  assert.deepEqual(calls.loginAzureRM, ["service-id"]);
  assert.equal(process.env.AZURE_CORE_OUTPUT, "json");
});

test("Input Validation Failures Do Not Clear Azure CLI Account", async () => {
  const codebasePath = makeProjectWithLog();
  const { task, calls } = loadTask({
    inputs: {
      azureSubscription: "service-id",
      deploymentName: "gpt-test",
      endpoint: "http://example.openai.azure.com/",
      apiVersion: "2025-03-01-preview",
      codebasePath,
      logFilePath: "logs/build.log",
      timeoutSeconds: "600",
    },
  });

  await task.run();

  assert.deepEqual(calls.setResult, [{
    result: "Failed",
    message: "endpoint must be a valid HTTPS URL: http://example.openai.azure.com/",
  }]);
  assert.equal(calls.spawnSync.some((call) => (
    call.command === "az" && Array.from(call.args || []).join(" ") === "account clear"
  )), false);
});

test("ServiceConnection Login Failures Are Properly Handled And Stop The Analyzer", async () => {
  const codebasePath = makeProjectWithLog();
  const { task, calls } = loadTask({
    inputs: {
      azureSubscription: "service-id",
      deploymentName: "gpt-test",
      endpoint: "https://example.openai.azure.com/",
      apiVersion: "2025-03-01-preview",
      codebasePath,
      logFilePath: "logs/build.log",
      timeoutSeconds: "600",
    },
    loginAzureRM: async () => { throw new Error("authentication failed"); },
  });

  await task.run();

  assert.deepEqual(calls.setResult, [{
    result: "Failed",
    message: "Azure service connection login failed for 'service-id': authentication failed",
  }]);
  assert.equal(calls.spawnSync.some((call) => (
    Array.from(call.args || [])[0] === path.join(taskDir, "log_analyzer_runner.py")
  )), false);
  const logoutCall = calls.spawnSync.at(-1);
  assert.equal(logoutCall.command, "az");
  assert.deepEqual(Array.from(logoutCall.args), ["account", "clear"]);
  assert.equal(logoutCall.options.stdio, "ignore");
});

test("Python Setup Command Failures Include Error Details", () => {
  const spawnError = loadTask({ spawnSync: () => ({ error: new Error("missing") }) });
  assert.throws(() => spawnError.task.runCommand("python3", ["--version"]), /Failed to run python3: missing/);

  const nonZero = loadTask({ spawnSync: () => ({ status: 2 }) });
  assert.throws(() => nonZero.task.runCommand("python3", ["-m", "venv"]), /exit 2/);
});

test("Output File Path Can Point To Missing Absolute File", () => {
  const { task } = loadTask();
  const codebasePath = makeProjectWithLog();
  const outputFilePath = path.join(codebasePath, "missing", "analysis.txt");
  const inputs = {
    codebasePath,
    logFilePath: "logs/build.log",
    endpoint: "https://example.openai.azure.com/",
    timeoutSeconds: "600",
    outputFilePath,
  };

  assert.equal(fs.existsSync(outputFilePath), false);
  task.validateInputs(inputs);
  assert.equal(inputs.outputFilePath, outputFilePath);
});

test("Output File Path Creates Directories And Replaces Existing File", async () => {
  const outputRoot = fs.mkdtempSync(path.join(os.tmpdir(), "microbots-output-test-"));
  const outputFilePath = path.join(outputRoot, "nested", "analysis.md");
  fs.mkdirSync(path.dirname(outputFilePath), { recursive: true });
  fs.writeFileSync(outputFilePath, "old analysis", "utf8");

  const { calls } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    inputs: { outputFilePath },
    mockMicrobots: { result: "New analysis\nwith ##vso[task.setvariable variable=x]unsafe-looking text" },
  });

  assert.equal(
    fs.readFileSync(outputFilePath, "utf8").replace(/\r\n/g, "\n"),
    "New analysis\nwith ##vso[task.setvariable variable=x]unsafe-looking text"
  );
});

test("Displayed LLM Output Does Not Emit Azure Pipelines Commands", async () => {
  const { calls, runnerResult } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    mockMicrobots: { result: "##vso[task.setvariable variable=NotAllowed]spoofed" },
  });

  assert.equal(runnerResult.status, 0, runnerResult.stderr);
  assert.match(runnerResult.stdout, / ##vso\[task\.setvariable variable=NotAllowed\]spoofed/);
  assert.equal(calls.setResult[0].result, "Succeeded");
});

test("Runner Reports Docker Sandbox Startup Failures", async () => {
  const { calls, events, runnerResult } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    mockMicrobots: { initError: "docker" },
  });

  assert.equal(events.some((event) => event.name === "loginAzureRM"), true);
  assert.equal(events.filter((event) => (
    event.name === "spawnSync" && Array.from(event.args || [])[0] === path.join(taskDir, "log_analyzer_runner.py")
  )).length, 1);
  assert.equal(runnerResult.status, 1);
  assert.match(runnerResult.stderr, /Docker-compatible daemon was not accessible/);
  assert.equal(calls.setResult[0].result, "Failed");
  assert.match(calls.setResult[0].message, /exit 1/);
});

test("Task Fails With Proper Error Message When LLM Deployment Cannot Be Reached (After Login With ServiceConnection)", async () => {
  const { calls, events, runnerResult } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    mockMicrobots: {
      status: false,
      result: "",
      error: "Deployment access failed",
    },
  });

  assert.equal(events.some((event) => event.name === "loginAzureRM"), true);
  assert.equal(runnerResult.status, 1);
  assert.match(runnerResult.stdout, /Deployment access failed/);
  assert.equal(calls.setResult[0].result, "Failed");
  assert.match(calls.setResult[0].message, /exit 1/);
});

test("Task Correctly Reports Failures While Analyzing Logs By The LLM", async () => {
  const { calls, runnerResult, runnerRecord } = await runTaskWithMockServiceConnectionAndMockMicrobots({
    mockMicrobots: {
      status: false,
      result: "",
      error: "Log analysis timed out",
    },
  });

  assert.equal(runnerResult.status, 1);
  assert.equal(runnerRecord.token, "mock-token");
  assert.match(runnerResult.stdout, /Log analysis timed out/);
  assert.equal(calls.setResult[0].result, "Failed");
  assert.match(calls.setResult[0].message, /exit 1/);
});
