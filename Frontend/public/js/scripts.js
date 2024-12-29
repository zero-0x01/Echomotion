// scripts.js

import * as THREE from '../three/build/three.module.js';
import { EffectComposer } from '../three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from '../three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from '../three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { io } from 'https://cdn.socket.io/4.5.4/socket.io.esm.min.js'; // Ensure you have access to Socket.IO client

// ----------------------
// 1. Variable Declarations
// ----------------------

// Audio-related variables
let audioContext;
let analyser;
let dataArray;
let source;

// Visualization control variables
let sharedStream = null; // To store the shared MediaStream
let mediaRecorder;
let audioChunks = [];
let isSendingAudio = false;
let isRecording = false; // To prevent overlapping recordings

// Target color for smooth transitions
let targetColor = { r: 0.5, g: 0.5, b: 0.5 }; // Initialize with a default color

// Clock for animation timing
const clock = new THREE.Clock(); // Ensure this is declared before animate()

// ----------------------
// 2. Setup Scene, Camera, Renderer
// ----------------------

// Select the container
const container = document.getElementById('shader-container');

// Renderer with transparent background
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setClearColor(0x000000, 0); // Fully transparent background
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// Scene and Camera
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
);
camera.position.set(0, -2, 14);
camera.lookAt(0, 0, 0);

// ----------------------
// 3. Shader and Mesh Setup
// ----------------------

// Uniforms for shader
const uniforms = {
    u_time: { value: 0.0 },
    u_frequency: { value: 0.0 },
    u_red: { value: 0.5 },
    u_green: { value: 0.5 },
    u_blue: { value: 0.5 },
};

// Shader material
const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: document.getElementById('vertexshader').textContent,
    fragmentShader: document.getElementById('fragmentshader').textContent,
    transparent: true,
});

// Geometry and mesh
const geometry = new THREE.IcosahedronGeometry(4, 30);
const mesh = new THREE.Mesh(geometry, material);
mesh.material.wireframe = true;
scene.add(mesh);

// ----------------------
// 4. Postprocessing Setup
// ----------------------

// Postprocessing with bloom effect
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(container.clientWidth, container.clientHeight),
    0.34,
    0.72,
    0.9
);
composer.addPass(renderPass);
composer.addPass(bloomPass);

// ----------------------
// 5. Socket.IO Setup
// ----------------------

// Socket.IO Client Setup
const socket = io('http://localhost:5000'); // Update with your server URL if different

// Handle connection
socket.on('connect', () => {
    console.log('Connected to WebSocket server');
});

// Handle emotion data
socket.on('emotion_data', (data) => {
    console.log('Received emotion data:', data);
    if (isSendingAudio) { // Update visualization only if audio sending is active
        updateVisualizationColors(data.emotions);
    }
});

// Handle disconnection
socket.on('disconnect', () => {
    console.log('Disconnected from WebSocket server');
});

// Function to send audio blob to the server
function sendAudioToServer(audioBlob) {
    return new Promise((resolve, reject) => {
        socket.emit('audio_data', audioBlob, (ack) => {
            if (ack === 'ok') {
                console.log('Audio data sent to server.');
                resolve();
            } else {
                console.error('Failed to send audio data to server.');
                reject('Failed to send audio data');
            }
        });
    });
}

// ----------------------
// 6. Animation Loop
// ----------------------

// Mouse movement for interaction
let mouseX = 0;
let mouseY = 0;
container.addEventListener('mousemove', (event) => {
    const bounds = container.getBoundingClientRect();
    mouseX = (event.clientX - bounds.left - bounds.width / 2) / 100;
    mouseY = (event.clientY - bounds.top - bounds.height / 2) / 100;
});

// Function to smoothly update colors based on targetColor
function updateColors(deltaTime) {
    uniforms.u_red.value += (targetColor.r - uniforms.u_red.value) * deltaTime * 0.5;
    uniforms.u_green.value += (targetColor.g - uniforms.u_green.value) * deltaTime * 0.5;
    uniforms.u_blue.value += (targetColor.b - uniforms.u_blue.value) * deltaTime * 0.5;
}

// Function to update visualization colors based on emotions
function updateVisualizationColors(emotions) {
    if (!emotions || Object.keys(emotions).length === 0) {
        // No emotions detected, do not update
        if (emotionDisplay) {
            emotionDisplay.textContent = '';
        }
        return;
    }

    // Sort emotions by percentage descending
    const sortedEmotions = Object.entries(emotions).sort((a, b) => b[1] - a[1]);

    // Get top emotion
    const topEmotion = sortedEmotions[0];
    if (emotionDisplay) {
        emotionDisplay.textContent = `Top Emotion: ${topEmotion[0]} (${(topEmotion[1] * 100).toFixed(2)}%)`;
    }

    // Map emotions to colors
    const emotionColorMap = {
        "Happy": { r: 1.0, g: 0.85, b: 0.1 }, // Yellow
        "Angry": { r: 1.0, g: 0.0, b: 0.0 }, // Red
        "Calm": { r: 0.0, g: 0.5, b: 1.0 },  // Blue
        "Sad": { r: 0.0, g: 0.0, b: 1.0 },    // Dark Blue
        "Surprised": { r: 1.0, g: 0.6, b: 0.0 },  // Orange
        "Fear": { r: 0.6, g: 0.0, b: 0.6 },       // Purple
        "Disgust": { r: 0.0, g: 1.0, b: 0.0 },    // Green
        "Neutral": { r: 0.5, g: 0.5, b: 0.5 },    // Grey
        // Add more emotions and colors as needed
    };

    // Calculate weighted average based on top 2 emotions
    let total = 0;
    let weightedColor = { r: 0, g: 0, b: 0 };
    sortedEmotions.slice(0, 2).forEach(([emotion, percentage]) => {
        const color = emotionColorMap[emotion] || { r: 0.5, g: 0.5, b: 0.5 }; // Default color if emotion not mapped
        weightedColor.r += color.r * percentage;
        weightedColor.g += color.g * percentage;
        weightedColor.b += color.b * percentage;
        total += percentage;
    });

    // Normalize the colors
    if (total > 0) {
        weightedColor.r /= total;
        weightedColor.g /= total;
        weightedColor.b /= total;
    } else {
        weightedColor = { r: 0.5, g: 0.5, b: 0.5 }; // Default color
    }

    // Update targetColor for smooth transition
    targetColor = weightedColor;
}

// Animation loop function with error handling
function animate() {
    requestAnimationFrame(animate);
    try {
        const deltaTime = clock.getDelta();

        uniforms.u_time.value += deltaTime * 0.5; // Adjust as needed

        // Update colors smoothly
        updateColors(deltaTime);

        // Update u_frequency based on audio
        if (analyser && dataArray) {
            analyser.getByteFrequencyData(dataArray);
            // Calculate average frequency
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            const avgFrequency = sum / dataArray.length;
            uniforms.u_frequency.value = avgFrequency;
        } else {
            uniforms.u_frequency.value = 0.0; // Reset frequency to zero when no audio
        }

        // Camera movement based on mouse
        camera.position.x += (mouseX - camera.position.x) * 0.05;
        camera.position.y += (-mouseY - camera.position.y) * 0.05;
        camera.lookAt(scene.position);

        renderer.clear();
        composer.render();
    } catch (error) {
        console.error("Error in animation loop:", error);
    }
}

// Start animation loop
animate();

// ----------------------
// 7. Audio Visualization Setup
// ----------------------

// Function to initialize Audio Context and Analyser
async function initAudioVisualization(stream) {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        source = audioContext.createMediaStreamSource(stream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        source.connect(analyser);
        console.log("Audio visualization initialized.");
    } catch (err) {
        console.error("Error initializing audio visualization:", err);
    }
}

// ----------------------
// 8. MediaRecorder Setup
// ----------------------

// Function to initialize MediaRecorder
async function initMediaRecorder(stream) {
    if (mediaRecorder) {
        console.warn("MediaRecorder is already initialized.");
        return;
    }
    try {
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            isRecording = false; // Recording has stopped
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];
            try {
                await sendAudioToServer(audioBlob);
                // Wait for backend response before sending the next clip
                setTimeout(() => {
                    if (isSendingAudio) {
                        startRecording();
                    }
                }, 1000); // 1-second pause after sending
            } catch (error) {
                console.error('Failed to send audio to server:', error);
                // Optionally, retry or notify the user
                setTimeout(() => {
                    if (isSendingAudio) {
                        startRecording();
                    }
                }, 1000); // 1-second pause before retrying
            }
        };

        mediaRecorder.onerror = (event) => {
            console.error("MediaRecorder error:", event.error);
        };

        console.log("MediaRecorder initialized.");
    } catch (err) {
        console.error("Error initializing MediaRecorder:", err);
    }
}

// Function to start recording
function startRecording() {
    if (mediaRecorder && mediaRecorder.state === 'inactive' && !isRecording) {
        mediaRecorder.start();
        isRecording = true;
        showRecordingIndicator();
        console.log("Recording started.");

        // Stop recording after 5 seconds
        setTimeout(() => {
            if (mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                console.log("Recording stopped.");
                hideRecordingIndicator();
                // 'onstop' will handle sending and starting the next recording
            }
        }, 5000); // 5 seconds
    }
}

// ----------------------
// 9. Audio Sending Control
// ----------------------

// Function to start sending audio clips
async function startSendingAudio() {
    if (isSendingAudio) {
        console.warn("Audio sending is already active.");
        return;
    }
    isSendingAudio = true;
    try {
        // Always get a new stream when starting
        sharedStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
        await initAudioVisualization(sharedStream);
        await initMediaRecorder(sharedStream);
        startRecording();
        console.log("Started sending audio clips to server.");
    } catch (err) {
        console.error("Error accessing microphone:", err);
        isSendingAudio = false;
        resetColor();
        sharedStream = null; // Ensure sharedStream is reset on error
    }
}

// Function to stop sending audio clips
function stopSendingAudio() {
    if (!isSendingAudio) {
        console.warn("Audio sending is not active.");
        return;
    }
    isSendingAudio = false;
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        console.log("Stopped sending audio clips to server.");
    }
    resetColor();
    // Optionally, stop the AudioContext to free resources
    if (audioContext) {
        audioContext.close();
        audioContext = null;
        analyser = null;
        source = null;
        dataArray = null;
        console.log("AudioContext closed.");
    }
    sharedStream = null; // Reset sharedStream
}

// Function to toggle audio sending
function toggleAudioSending() {
    if (!isSendingAudio) {
        startSendingAudio();
    } else {
        stopSendingAudio();
    }
}

// Add click event listener to toggle audio sending
container.addEventListener('click', toggleAudioSending);

// ----------------------
// 10. Visual Indicators Setup
// ----------------------

// Select DOM elements for visual indicators
const recordingIndicator = document.getElementById('recording-indicator');
const emotionDisplay = document.getElementById('emotion-display');

// Function to show recording indicator
function showRecordingIndicator() {
    if (recordingIndicator) {
        recordingIndicator.style.display = 'block';
    }
}

// Function to hide recording indicator
function hideRecordingIndicator() {
    if (recordingIndicator) {
        recordingIndicator.style.display = 'none';
    }
}

// ----------------------
// 11. Reset Function
// ----------------------

// Function to reset colors and frequency
function resetColor() {
    targetColor = { r: 0.5, g: 0.5, b: 0.5 };
    uniforms.u_red.value = targetColor.r;
    uniforms.u_green.value = targetColor.g;
    uniforms.u_blue.value = targetColor.b;
    uniforms.u_frequency.value = 0.0; // Reset frequency to zero
}

// ----------------------
// 12. Window Resize Handling
// ----------------------

// Optional: Handle window resize to adjust renderer and camera
window.addEventListener('resize', () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    composer.setSize(width, height);
});
