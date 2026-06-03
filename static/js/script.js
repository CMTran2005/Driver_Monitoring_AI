// ==========================================
// CÀI ĐẶT - LOAD TỪ SERVER
// ==========================================
let isSoundEnabled = true;

async function loadSettings() {
    try {
        const res = await fetch('/api/settings');
        const s = await res.json();
        isSoundEnabled = s.soundEnabled !== false;

        if (window.location.pathname.includes('settings')) {
            const toggle   = document.getElementById('soundToggle');
            const senRange = document.getElementById('sensitivityRange');
            const senVal   = document.getElementById('senValue');
            const earRange = document.getElementById('earThreshRange');
            const earVal   = document.getElementById('earThreshValue');
            const marRange = document.getElementById('marThreshRange');
            const marVal   = document.getElementById('marThreshValue');

            if (toggle)   toggle.checked   = s.soundEnabled !== false;
            if (senRange) { senRange.value = s.sensitivity || 2.0; senVal.innerText = s.sensitivity || 2.0; }
            if (earRange) { earRange.value = s.earThresh   || 0.22; earVal.innerText = (s.earThresh || 0.22).toFixed(2); }
            if (marRange) { marRange.value = s.marThresh   || 0.60; marVal.innerText = (s.marThresh || 0.60).toFixed(2); }
        }
    } catch(e) {}
}

loadSettings();

document.getElementById('sensitivityRange')?.addEventListener('input', function() {
    document.getElementById('senValue').innerText = this.value;
});
document.getElementById('earThreshRange')?.addEventListener('input', function() {
    document.getElementById('earThreshValue').innerText = parseFloat(this.value).toFixed(2);
});
document.getElementById('marThreshRange')?.addEventListener('input', function() {
    document.getElementById('marThreshValue').innerText = parseFloat(this.value).toFixed(2);
});

document.getElementById('saveSettingsBtn')?.addEventListener('click', async function() {
    const sensitivity  = parseFloat(document.getElementById('sensitivityRange').value);
    const earThresh    = parseFloat(document.getElementById('earThreshRange').value);
    const marThresh    = parseFloat(document.getElementById('marThreshRange').value);
    const soundEnabled = document.getElementById('soundToggle').checked;
    const eyeFrames    = Math.round(sensitivity * 15);
    const yawnFrames   = 15; // cố định nhanh ~1 giây

    try {
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ soundEnabled, sensitivity, earThresh, marThresh, eyeFrames, yawnFrames })
        });
        const data = await res.json();
        if (data.ok) {
            isSoundEnabled = soundEnabled;
            const toast = document.getElementById('save-toast');
            if (toast) { toast.classList.remove('d-none'); setTimeout(() => toast.classList.add('d-none'), 3000); }
        }
    } catch(e) { alert("❌ Lỗi kết nối server."); }
});

// ==========================================
// CẢNH BÁO
// ==========================================
const robotVoice = new SpeechSynthesisUtterance();
robotVoice.lang = 'vi-VN';
robotVoice.text = "Cảnh báo nguy hiểm. Phát hiện tài xế đang buồn ngủ. Vui lòng hãy tập trung!";
let isAlerting = false;
let lastAlertStatus = "tỉnh táo";

function triggerAlert(type) {
    if (isAlerting && lastAlertStatus === type) return;
    isAlerting = true;
    lastAlertStatus = type;

    const statusText = document.getElementById("status-text");
    const alertBox   = document.getElementById("alert-box");
    const alertMsg   = document.getElementById("alert-msg");

    const label = type === "buồn ngủ" ? "BUỒN NGỦ" : type === "đang ngáp" ? "ĐANG NGÁP" : "MẤT TẬP TRUNG";
    const cls   = type === "buồn ngủ" ? "text-danger" : "text-warning";
    const msg   = type === "buồn ngủ" ? "⚠️ PHÁT HIỆN BUỒN NGỦ!" : "⚠️ PHÁT HIỆN NGÁP!";

    if (statusText) { statusText.innerText = label; statusText.className = "fw-bold " + cls; }
    if (alertBox)   { alertBox.classList.remove("d-none"); alertBox.classList.add("alert-active"); }
    if (alertMsg)   alertMsg.innerText = msg;

    if (isSoundEnabled && !window.speechSynthesis.speaking) {
        window.speechSynthesis.speak(robotVoice);
    }
}

function stopAlert() {
    isAlerting = false;
    lastAlertStatus = "tỉnh táo";
    const statusText = document.getElementById("status-text");
    const alertBox   = document.getElementById("alert-box");
    if (statusText) { statusText.innerText = "TỈNH TÁO"; statusText.className = "text-success fw-bold"; }
    if (alertBox)   { alertBox.classList.add("d-none"); alertBox.classList.remove("alert-active"); }
    window.speechSynthesis.cancel();
}

// ==========================================
// BIỂU ĐỒ EAR REAL-TIME
// ==========================================
const chartElement = document.getElementById('earChart');

if (chartElement) {
    const MAX_PTS = 40;
    const ctx = chartElement.getContext('2d');
    const earChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(MAX_PTS).fill(''),
            datasets: [{
                label: 'EAR thực tế',
                data: Array(MAX_PTS).fill(0.30),
                borderColor: '#00ff99',
                backgroundColor: 'rgba(0,255,153,0.07)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }, {
                label: 'Ngưỡng nguy hiểm',
                data: Array(MAX_PTS).fill(0.22),
                borderColor: '#ff4444',
                borderWidth: 1.5,
                borderDash: [6, 4],
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            animation: false,
            plugins: { legend: { labels: { color: '#ccc', font: { size: 12 } } } },
            scales: {
                y: { min: 0.05, max: 0.45, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#aaa' } },
                x: { display: false }
            }
        }
    });

    setInterval(async () => {
        try {
            const res  = await fetch('/api/status');
            const data = await res.json();
            const ear    = parseFloat(data.ear) || 0.30;
            const status = data.status;

            earChart.data.datasets[0].data.push(ear);
            earChart.data.datasets[0].data.shift();
            earChart.data.labels.push('');
            earChart.data.labels.shift();
            earChart.data.datasets[0].borderColor     = ear < 0.22 ? '#ff4444' : '#00ff99';
            earChart.data.datasets[0].backgroundColor = ear < 0.22 ? 'rgba(255,68,68,0.1)' : 'rgba(0,255,153,0.07)';
            earChart.update();

            const earVal = document.getElementById('ear-value');
            if (earVal) earVal.innerText = ear.toFixed(3);

            if (status !== "tỉnh táo") triggerAlert(status);
            else if (isAlerting) stopAlert();

        } catch(e) {}
    }, 500);
}

// ==========================================
// LỊCH SỬ - hàm ở global scope để onclick gọi được
// ==========================================
async function loadHistory() {
    try {
        const res   = await fetch('/api/history');
        const data  = await res.json();
        const tbody = document.getElementById('history-tbody');
        if (!tbody) return;

        const lastUpdate = document.getElementById('last-update');
        if (lastUpdate) lastUpdate.innerText = 'Cập nhật: ' + new Date().toLocaleTimeString('vi-VN');

        if (data.length === 0) {
            tbody.innerHTML = `
                <tr><td colspan="4" class="text-center text-white py-5">
                    <i class="bi bi-check-circle text-success fs-4"></i><br>
                    Chưa có cảnh báo nào trong phiên này.
                </td></tr>`;
            return;
        }

        tbody.innerHTML = data.map((item, i) => `
            <tr>
                <td class="text-muted">${i + 1}</td>
                <td><i class="bi bi-clock me-1 text-secondary"></i>${item.time}</td>
                <td>${item.type}</td>
                <td><span class="badge bg-${item.level}">${item.level_text}</span></td>
            </tr>
        `).join('');
    } catch(e) {
        const tbody = document.getElementById('history-tbody');
        if (tbody) tbody.innerHTML = `
            <tr><td colspan="4" class="text-center text-danger py-3">
                ❌ Không thể kết nối server.
            </td></tr>`;
    }
}

async function clearHistory() {
    if (!confirm("Xóa toàn bộ lịch sử cảnh báo?")) return;
    try {
        await fetch('/api/history/clear', { method: 'POST' });
        loadHistory();
    } catch(e) {}
}

if (window.location.pathname.includes('history')) {
    loadHistory();
    setInterval(loadHistory, 5000);
}