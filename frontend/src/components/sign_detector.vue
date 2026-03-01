<template>
  <div class="container">
    <div class="side-panel">
      <h2>Sign Language AI</h2>
      
      <div class="result-card">
        <span class="status" :class="{ 'active': isVisible }">
          {{ isVisible ? '• Live' : '• No Hands' }}
        </span>
        <div class="label-display">{{ prediction.label }}</div>
        <div class="confidence-meter">
          <div class="bar" :style="{ width: (prediction.confidence * 100) + '%' }"></div>
        </div>
        <small>Confidence: {{ (prediction.confidence * 100).toFixed(1) }}%</small>
      </div>
    </div>

    <div class="video-wrap">
      <video ref="videoRef" style="display:none"></video>
      <canvas ref="canvasRef" width="640" height="480"></canvas>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { Holistic } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';
import * as drawing_utils from '@mediapipe/drawing_utils';
import * as mp_holistic from '@mediapipe/holistic';
import axios from 'axios';

const videoRef = ref(null);
const canvasRef = ref(null);
const isVisible = ref(false);
const prediction = ref({ label: 'Waiting...', confidence: 0 });

// Throttle configuration
let lastTime = 0;
const THROTTLE_MS = 200; 

onMounted(() => {
  const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });

  holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  holistic.onResults(async (results) => {
    draw(results);
    
    // Check if hands are in frame
    isVisible.value = !!(results.leftHandLandmarks || results.rightHandLandmarks);

    const now = Date.now();
    if (isVisible.value && (now - lastTime > THROTTLE_MS)) {
      lastTime = now;
      const landmarks = formatLandmarks(results);
      sendRequest(landmarks);
    }
  });

  const camera = new Camera(videoRef.value, {
    onFrame: async () => { await holistic.send({ image: videoRef.value }); },
    width: 640, height: 480
  });
  camera.start();
});

const formatLandmarks = (res) => {
  const pull = (pts, len) => pts ? pts.map(p => [p.x, p.y, p.z]) : Array(len).fill([0, 0, 0]);
  return [
    ...pull(res.faceLandmarks, 468),
    ...pull(res.leftHandLandmarks, 21),
    ...pull(res.poseLandmarks, 33),
    ...pull(res.rightHandLandmarks, 21)
  ];
};

const sendRequest = async (landmarks) => {
  try {
    const res = await axios.post('http://localhost:5000/predict', { landmarks });
    prediction.value = {
      label: res.data.label,
      confidence: res.data.confidence
    };
  } catch (e) {
    console.error("Backend Error", e);
  }
};

const draw = (res) => {
  const ctx = canvasRef.value.getContext('2d');
  ctx.save();
  ctx.clearRect(0, 0, 640, 480);
  ctx.drawImage(res.image, 0, 0, 640, 480);
  
  // Draw Face, Hands, and Pose
  drawing_utils.drawConnectors(ctx, res.leftHandLandmarks, mp_holistic.HAND_CONNECTIONS, {color: '#FF0000', lineWidth: 2});
  drawing_utils.drawConnectors(ctx, res.rightHandLandmarks, mp_holistic.HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
  drawing_utils.drawConnectors(ctx, res.poseLandmarks, mp_holistic.POSE_CONNECTIONS, {color: '#FFFFFF', lineWidth: 1});
  ctx.restore();
};
</script>

<style scoped>
.container { display: flex; background: #0f172a; min-height: 100vh; color: white; font-family: sans-serif; padding: 20px; gap: 20px; }
.side-panel { width: 300px; }
.video-wrap { flex: 1; background: #000; border-radius: 12px; overflow: hidden; display: flex; justify-content: center; align-items: center; border: 2px solid #334155; }
canvas { max-width: 100%; height: auto; }
.result-card { background: #1e293b; padding: 20px; border-radius: 12px; margin-top: 20px; border: 1px solid #334155; }
.label-display { font-size: 2.5rem; font-weight: bold; color: #38bdf8; margin: 10px 0; }
.status { font-size: 0.8rem; color: #64748b; }
.status.active { color: #4ade80; }
.confidence-meter { background: #334155; height: 6px; border-radius: 3px; margin: 10px 0; overflow: hidden; }
.bar { background: #38bdf8; height: 100%; transition: width 0.3s; }
</style>