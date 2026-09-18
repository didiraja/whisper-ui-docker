const form = document.querySelector("#transcription-form");
const sourceYoutube = document.querySelector("#source-youtube");
const sourceUpload = document.querySelector("#source-upload");
const youtubePanel = document.querySelector("#youtube-panel");
const uploadPanel = document.querySelector("#upload-panel");
const youtubeUrl = document.querySelector("#youtube-url");
const audioFile = document.querySelector("#audio-file");
const uploadDropZone = document.querySelector("#upload-drop-zone");
const fileSelection = document.querySelector("#file-selection");
const formError = document.querySelector("#form-error");
const jobPanel = document.querySelector("#job-panel");
const jobStage = document.querySelector("#job-stage");
const jobPipeline = document.querySelector("#job-pipeline");
const jobStepper = document.querySelector("#job-stepper");
const jobElapsed = document.querySelector("#job-elapsed");
const jobRemaining = document.querySelector("#job-remaining");
const jobProgress = document.querySelector("#job-progress");
const jobLogLines = document.querySelector("#job-log-lines");
const jobWarning = document.querySelector("#job-warning");
const resultPanel = document.querySelector("#result-panel");
const transcript = document.querySelector("#transcript");
const copyButton = document.querySelector("#copy-transcript");
const downloadTxt = document.querySelector("#download-txt");
const downloadSrt = document.querySelector("#download-srt");
const resetSessionButton = document.querySelector("#reset-session");
let pollTimer = null;
let requestGeneration = 0;
let pollFailures = 0;
const normalPollDelay = 1500;
const maximumRetryDelay = 10000;
const stepElements = [...jobStepper.querySelectorAll("li")];
const stepIndexByStage = {
  loading_model: 0,
  transcribing: 1,
  aligning: 2,
  formatting: 3,
  complete: 3,
};
const progressLabels = {
  transcribing: "Transcription progress",
  aligning: "Alignment progress",
};

const stageLabels = {
  validating: "Validating input…",
  receiving: "Receiving upload…",
  downloading: "Downloading YouTube audio…",
  preparing: "Preparing audio…",
  loading_model: "Downloading or loading model…",
  transcribing: "Transcribing…",
  aligning: "Aligning subtitles…",
  formatting: "Preparing downloads…",
  complete: "Transcription complete",
};

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    const error = new Error("The server could not be reached.");
    error.code = "network_error";
    error.status = null;
    throw error;
  }
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error?.message || "Request failed.");
    error.code = body.error?.code || "request_failed";
    error.status = response.status;
    throw error;
  }
  return body;
}

function showError(message) {
  formError.textContent = message || "Something went wrong.";
  formError.hidden = false;
}

function clearError() {
  formError.textContent = "";
  formError.hidden = true;
}

function stopPolling() {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function beginJobFlow() {
  stopPolling();
  pollFailures = 0;
  requestGeneration += 1;
  return requestGeneration;
}

function isCurrentFlow(generation) {
  return generation === requestGeneration;
}

function clearResult() {
  resultPanel.hidden = true;
  transcript.value = "";
  downloadTxt.removeAttribute("href");
  downloadSrt.removeAttribute("href");
}

function clearPipeline() {
  jobPipeline.hidden = true;
  jobElapsed.textContent = "";
  jobRemaining.textContent = "";
  jobProgress.hidden = true;
  jobProgress.removeAttribute("value");
  jobLogLines.replaceChildren();
}

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  if (hours) return `${hours}h ${minutes}m ${remainder}s`;
  if (minutes) return `${minutes}m ${remainder}s`;
  return `${remainder}s`;
}

function renderPipeline(job) {
  const pipeline = job.pipeline || {};
  if (!Number.isFinite(pipeline.elapsed_seconds)) {
    clearPipeline();
    return;
  }

  jobPipeline.hidden = false;
  const currentIndex = stepIndexByStage[job.stage] ?? 0;
  const progress = Number.isFinite(pipeline.progress_percent)
    ? pipeline.progress_percent
    : null;

  for (const [index, step] of stepElements.entries()) {
    const marker = step.querySelector(".step-marker");
    const status = step.querySelector(".step-status");
    const isComplete = job.state === "completed" || index < currentIndex;
    const isCurrent = !isComplete && index === currentIndex;
    const isFailed = isCurrent && job.state === "failed";
    const isFallback = index === 2 && pipeline.alignment_fallback && isComplete;
    step.classList.toggle("is-complete", isComplete);
    step.classList.toggle("is-current", isCurrent && !isFailed);
    step.classList.toggle("is-failed", isFailed);
    step.classList.toggle("is-warning", isFallback);
    if (isCurrent && !isFailed) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");

    marker.textContent = isFallback
      ? "!"
      : (isComplete ? "✓" : (isFailed ? "!" : String(index + 1)));
    if (isFallback) status.textContent = "Fallback";
    else if (isComplete) status.textContent = "Done";
    else if (isFailed) status.textContent = "Stopped";
    else if (isCurrent && progress !== null) status.textContent = `${Math.round(progress)}%`;
    else if (isCurrent) status.textContent = "In progress";
    else status.textContent = "Waiting";
  }

  const elapsed = formatDuration(pipeline.elapsed_seconds);
  if (job.state === "completed") jobElapsed.textContent = `Finished in ${elapsed}`;
  else if (job.state === "failed") jobElapsed.textContent = `Stopped after ${elapsed}`;
  else jobElapsed.textContent = `Elapsed ${elapsed}`;

  if (job.state === "completed") jobRemaining.textContent = "All steps complete";
  else if (job.state === "failed") jobRemaining.textContent = "No remaining-time estimate";
  else if (Number.isFinite(pipeline.estimated_remaining_seconds)) {
    jobRemaining.textContent = `About ${formatDuration(pipeline.estimated_remaining_seconds)} remaining in this step`;
  } else {
    jobRemaining.textContent = "Estimating current step…";
  }

  const showProgress = job.state === "running" && progress !== null;
  jobProgress.hidden = !showProgress;
  if (showProgress) {
    jobProgress.value = progress;
    jobProgress.setAttribute(
      "aria-label",
      progressLabels[job.stage] || "Current step progress",
    );
  } else jobProgress.removeAttribute("value");

  const lines = Array.isArray(pipeline.log_lines)
    ? pipeline.log_lines.slice(-5)
    : [];
  jobLogLines.replaceChildren(...lines.map((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    return item;
  }));
}

function schedulePoll(jobId, generation, delay = normalPollDelay) {
  if (!isCurrentFlow(generation)) return;
  stopPolling();
  pollTimer = window.setTimeout(() => {
    pollTimer = null;
    pollJob(jobId, generation).catch((error) => {
      showPollingError(error, jobId, generation);
    });
  }, delay);
}

function showPollingError(error, jobId, generation) {
  if (!isCurrentFlow(generation)) return;
  stopPolling();
  if (jobId && (error.status === 404 || error.code === "not_found")) {
    jobStage.textContent = "Checking the current job…";
    reconcileMissingJob(generation).catch((reconcileError) => {
      showPollingError(reconcileError, null, generation);
    });
    return;
  }
  pollFailures += 1;
  const retryDelay = Math.min(
    normalPollDelay * (2 ** Math.min(pollFailures - 1, 4)),
    maximumRetryDelay,
  );
  jobPanel.hidden = false;
  jobStage.textContent = "Connection interrupted; retrying status…";
  showError(`${error.message || "Status could not be refreshed."} Retrying automatically.`);
  setBusy(true);
  schedulePoll(jobId, generation, retryDelay);
}

async function reconcileMissingJob(generation) {
  const job = await requestJson("/api/jobs/current");
  if (!isCurrentFlow(generation)) return;
  pollFailures = 0;
  clearError();
  if (!job) {
    stopPolling();
    clearResult();
    clearPipeline();
    jobPanel.hidden = false;
    jobStage.textContent = "The previous job is no longer available.";
    jobWarning.textContent = "";
    jobWarning.hidden = true;
    setBusy(false);
    return;
  }
  renderJob(job);
  if (job.state === "accepted" || job.state === "running") {
    schedulePoll(job.id, generation);
  } else {
    stopPolling();
  }
}

async function submitJob(event) {
  event.preventDefault();
  const generation = beginJobFlow();
  clearError();
  clearResult();
  clearPipeline();
  jobPanel.hidden = false;
  jobWarning.textContent = "";
  jobWarning.hidden = true;
  const payload = new FormData(form);
  setBusy(true);
  jobStage.textContent = sourceUpload.checked ? "Receiving upload…" : "Submitting URL…";
  try {
    const job = await requestJson("/api/jobs", { method: "POST", body: payload });
    if (!isCurrentFlow(generation)) return;
    renderJob(job);
    pollJob(job.id, generation).catch((error) => {
      showPollingError(error, job.id, generation);
    });
  } catch (error) {
    if (!isCurrentFlow(generation)) return;
    stopPolling();
    showError(error.message);
    setBusy(false);
  }
}

async function pollJob(jobId, generation) {
  const job = await requestJson(jobId ? `/api/jobs/${jobId}` : "/api/jobs/current");
  if (!isCurrentFlow(generation)) return;
  pollFailures = 0;
  clearError();
  if (!job) {
    jobPanel.hidden = true;
    clearPipeline();
    setBusy(false);
    return;
  }
  renderJob(job);
  if (job.state === "accepted" || job.state === "running") {
    schedulePoll(job.id, generation);
  } else {
    stopPolling();
  }
}

function renderJob(job) {
  jobPanel.hidden = false;
  jobStage.textContent = stageLabels[job.stage] || job.stage;
  renderPipeline(job);
  jobWarning.textContent = job.warning || "";
  jobWarning.hidden = !job.warning;
  if (job.state === "completed") {
    transcript.value = job.transcript || "";
    downloadTxt.href = job.downloads.txt;
    downloadSrt.href = job.downloads.srt;
    resetSessionButton.disabled = false;
    resultPanel.hidden = false;
    setBusy(false);
  } else if (job.state === "failed") {
    clearResult();
    showError(job.error);
    setBusy(false);
  } else {
    clearResult();
    setBusy(true);
  }
}

function setBusy(isBusy) {
  for (const control of form.elements) control.disabled = isBusy;
  if (!isBusy) updateSourcePanels();
  uploadDropZone.classList.toggle("is-disabled", audioFile.disabled);
  uploadDropZone.setAttribute("aria-disabled", String(audioFile.disabled));
}

function updateFileSelection() {
  fileSelection.textContent = audioFile.files.length
    ? audioFile.files[0].name
    : "No file selected";
}

function updateSourcePanels() {
  const uploadSelected = sourceUpload.checked;
  uploadPanel.hidden = !uploadSelected;
  youtubePanel.hidden = uploadSelected;
  audioFile.disabled = !uploadSelected;
  youtubeUrl.disabled = uploadSelected;
  if (uploadSelected) youtubeUrl.value = "";
  else {
    audioFile.value = "";
    updateFileSelection();
  }
  uploadDropZone.classList.toggle("is-disabled", audioFile.disabled);
  uploadDropZone.setAttribute("aria-disabled", String(audioFile.disabled));
}

async function recoverCurrentJob(generation) {
  await pollJob(null, generation);
}

async function copyTranscript() {
  try {
    await navigator.clipboard.writeText(transcript.value);
  } catch {
    transcript.select();
    document.execCommand("copy");
  }
}

async function resetSession() {
  const generation = beginJobFlow();
  resetSessionButton.disabled = true;
  clearError();
  try {
    await requestJson("/api/jobs/current", { method: "DELETE" });
    if (!isCurrentFlow(generation)) return;
    clearResult();
    clearPipeline();
    jobPanel.hidden = true;
    jobWarning.textContent = "";
    jobWarning.hidden = true;
    form.reset();
    resetSessionButton.disabled = false;
    updateFileSelection();
    updateSourcePanels();
  } catch (error) {
    if (!isCurrentFlow(generation)) return;
    resetSessionButton.disabled = false;
    showError(error.message);
  }
}

sourceYoutube.addEventListener("change", updateSourcePanels);
sourceUpload.addEventListener("change", updateSourcePanels);
audioFile.addEventListener("change", updateFileSelection);
for (const eventName of ["dragenter", "dragover"]) {
  uploadDropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (audioFile.disabled) return;
    event.dataTransfer.dropEffect = "copy";
    uploadDropZone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "dragend"]) {
  uploadDropZone.addEventListener(eventName, () => {
    uploadDropZone.classList.remove("is-dragging");
  });
}
uploadDropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadDropZone.classList.remove("is-dragging");
  if (audioFile.disabled) return;
  const files = event.dataTransfer.files;
  if (files.length !== 1 || !/\.(mp3|wav)$/i.test(files[0].name)) {
    audioFile.value = "";
    updateFileSelection();
    showError("Drop one MP3 or WAV file.");
    return;
  }
  try {
    audioFile.files = files;
    clearError();
    updateFileSelection();
  } catch {
    showError("This browser cannot attach a dropped file; use the file chooser.");
  }
});
form.addEventListener("submit", submitJob);
copyButton.addEventListener("click", copyTranscript);
resetSessionButton.addEventListener("click", resetSession);
window.addEventListener("DOMContentLoaded", () => {
  updateSourcePanels();
  const generation = beginJobFlow();
  recoverCurrentJob(generation).catch((error) => {
    showPollingError(error, null, generation);
  });
});
