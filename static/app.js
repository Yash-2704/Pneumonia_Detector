const API_URL = "/predict";

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const diagnosisText = document.getElementById("diagnosis-text");
const probabilityText = document.getElementById("probability-text");
const confidenceText = document.getElementById("confidence-text");
const timeText = document.getElementById("time-text");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#dc2626" : "#475569";
}

function renderResult(data) {
  diagnosisText.textContent = data.diagnosis.toUpperCase();
  probabilityText.textContent = `${(data.probability * 100).toFixed(2)}%`;
  confidenceText.textContent = `${(data.confidence * 100).toFixed(2)}%`;
  timeText.textContent = data.inference_time_ms.toFixed(2);

  diagnosisText.parentElement.classList.toggle("warning", data.diagnosis === "pneumonia");
  diagnosisText.parentElement.classList.toggle("success", data.diagnosis === "normal");

  resultEl.classList.remove("hidden");
}

fileInput.addEventListener("change", () => {
  if (!fileInput.files || !fileInput.files[0]) {
    fileName.textContent = "Choose an image...";
    return;
  }
  fileName.textContent = fileInput.files[0].name;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files || !fileInput.files[0]) {
    setStatus("Please select an X-ray image first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  submitBtn.disabled = true;
  setStatus("Running inference...");
  resultEl.classList.add("hidden");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || error.detail || "Server error");
    }

    const data = await response.json();
    renderResult(data);
    setStatus("Inference complete.");
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Prediction failed.", true);
  } finally {
    submitBtn.disabled = false;
  }
});
