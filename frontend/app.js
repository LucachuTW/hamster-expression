const API_URL = "http://localhost:8000/api/predict/stream";
const POLL_INTERVAL_MS = 1000;

const emotionImages = {
  angry: "assets/emotions/angry.png",
  contempt: "assets/emotions/contempt.png",
  disgust: "assets/emotions/disgust.png",
  fear: "assets/emotions/fear.png",
  happy: "assets/emotions/happy.png",
  natural: "assets/emotions/natural.png",
  sad: "assets/emotions/sad.png",
  sleepy: "assets/emotions/sleepy.png",
  surprised: "assets/emotions/surprised.png",
  unknown: "assets/emotions/unknown.png",
};

const video = document.getElementById("camera");
const canvas = document.getElementById("frame-canvas");
const statusLabel = document.getElementById("status");
const emotionLabel = document.getElementById("emotion-label");
const confidenceLabel = document.getElementById("confidence-label");
const emotionImage = document.getElementById("emotion-image");

let stream = null;
let lastPrediction = "unknown";
let isSending = false;

async function initCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    statusLabel.textContent = "Camera started. Sending frames to backend...";
  } catch (error) {
    console.error("Camera error:", error);
    statusLabel.textContent = "Unable to access camera. Please allow permissions.";
  }
}

function captureFrame() {
  if (!stream) return null;
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    return null;
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.7);
}

async function requestPrediction() {
  if (isSending) return;
  const frame = captureFrame();
  if (!frame) return;
  isSending = true;
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: frame }),
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    const data = await response.json();
    updateUI(data);
    statusLabel.textContent = "Streaming predictions...";
  } catch (error) {
    console.error("Prediction error:", error);
    statusLabel.textContent = "Prediction failed. Check backend logs.";
  } finally {
    isSending = false;
  }
}

function updateUI(prediction) {
  const emotion = prediction.emotion || "unknown";
  const confidence = prediction.confidence || 0;
  lastPrediction = emotion;
  emotionLabel.textContent = `Detected: ${emotion}`;
  confidenceLabel.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
  const imagePath = emotionImages[emotion.toLowerCase()] || emotionImages.unknown;
  emotionImage.src = imagePath;
}

function startPolling() {
  setInterval(requestPrediction, POLL_INTERVAL_MS);
}

document.addEventListener("DOMContentLoaded", async () => {
  await initCamera();
  startPolling();
});
