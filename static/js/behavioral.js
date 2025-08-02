const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const stopBtn = document.getElementById('stopBtn');
const metricsDiv = document.getElementById('metrics');
const resultDiv = document.getElementById('resultLinks');
const ctx = canvas.getContext('2d');

const socket = io();

// === Start webcam ===
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        drawLoop();
    })
    .catch(err => alert('Error accessing webcam: ' + err));

function drawLoop() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            socket.emit('frame', reader.result);
        };
        reader.readAsDataURL(blob);
    }, 'image/jpeg');
    requestAnimationFrame(drawLoop);
}

socket.on('metrics', data => {
    let html = '';
    for (let key in data) {
        html += `<b>${key}:</b> ${data[key]}<br>`;
    }
    metricsDiv.innerHTML = html;
});

socket.on('allow_stop', () => {
    stopBtn.disabled = false;
});

stopBtn.addEventListener('click', () => {
    socket.emit('stop_behavioral');
});
socket.on('not_enough_samples', ({ count }) => {
    alert(`Only ${count} samples collected. Please wait for more.`);
});

socket.on('behavioral_done', ({ summary, csv_path, pdf_path }) => {
    metricsDiv.innerHTML = '<h3>Final Summary</h3>';
    for (let key in summary) {
        metricsDiv.innerHTML += `<b>${key}:</b> ${summary[key]}<br>`;
    }
    resultDiv.innerHTML = `
        <a href="/download/${csv_path}" target="_blank">📁 Download CSV</a><br>
        <a href="/download/${pdf_path}" target="_blank">📄 Download PDF Report</a><br>
        <a href="/dashboard"><button>Back to Dashboard</button></a>
    `;
    stopBtn.disabled = true;
});

