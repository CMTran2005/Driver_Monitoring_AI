// ==========================================
// XỬ LÝ LƯU VÀ ĐỌC CÀI ĐẶT (LOCAL STORAGE)
// ==========================================

// 1. Đọc cài đặt cũ (nếu chưa có thì lấy mặc định là Bật âm thanh, độ nhạy 2.0s)
let isSoundEnabled = localStorage.getItem('soundEnabled') !== 'false';
let currentSensitivity = localStorage.getItem('sensitivity') || 2.0;

// 2. Nếu đang ở trang Cài Đặt, tự động điền giá trị đã lưu vào form
if (window.location.pathname.includes('settings.html')) {
    document.getElementById('soundToggle').checked = isSoundEnabled;
    document.getElementById('sensitivityRange').value = currentSensitivity;
    document.getElementById('senValue').innerText = currentSensitivity;

    // Khi bấm nút "Lưu Cài đặt"
    document.getElementById('saveSettingsBtn').addEventListener('click', function () {
        const soundStatus = document.getElementById('soundToggle').checked;
        const senStatus = document.getElementById('sensitivityRange').value;

        // Lưu vào bộ nhớ trình duyệt
        localStorage.setItem('soundEnabled', soundStatus);
        localStorage.setItem('sensitivity', senStatus);

        alert("✅ Đã lưu cài đặt thành công! Hệ thống sẽ áp dụng ngay.");
    });
}
// Tạo một đối tượng giọng nói ảo
const robotVoice = new SpeechSynthesisUtterance();
robotVoice.lang = 'vi-VN'; // Chọn tiếng Việt và 
robotVoice.text = "Cảnh báo nguy hiểm. Phát hiện tài xế đang buồn ngủ. Vui lòng hãy tập trung!";

function triggerAlert() {
    const statusText = document.getElementById("status-text");
    const alertBox = document.getElementById("alert-box");

    statusText.innerText = "BUỒN NGỦ / MẤT TẬP TRUNG";
    statusText.className = "text-danger fw-bold";

    alertBox.classList.remove("d-none");
    alertBox.classList.add("alert-active");

    // KIỂM TRA CÀI ĐẶT TRƯỚC KHI BÁO ĐỘNG
    if (isSoundEnabled) {
        window.speechSynthesis.speak(robotVoice);
    } else {
        console.log("Cảnh báo bằng hình ảnh (Đã tắt âm thanh trong cài đặt)");
    }
}

function stopAlert() {
    const statusText = document.getElementById("status-text");
    const alertBox = document.getElementById("alert-box");

    statusText.innerText = "TỈNH TÁO";
    statusText.className = "text-success fw-bold";

    alertBox.classList.add("d-none");
    alertBox.classList.remove("alert-active");

    // Bấm tắt thì ngừng nói
    window.speechSynthesis.cancel();
}
// ==========================================
// CODE BIỂU ĐỒ EAR THỜI GIAN THỰC (CHART.JS)
// ==========================================

// ==========================================
// CODE BIỂU ĐỒ EAR THỜI GIAN THỰC (CHART.JS)
// ==========================================

// Tìm thẻ biểu đồ trước
const chartElement = document.getElementById('earChart');

// CHỈ VẼ BIỂU ĐỒ NẾU ĐANG Ở TRANG GIÁM SÁT (TỒN TẠI THẺ EARCHART)
if (chartElement) {
    const ctx = chartElement.getContext('2d');
    const earChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            datasets: [{
                label: 'Độ mở của mắt (EAR)',
                data: [0.3, 0.32, 0.31, 0.3, 0.33, 0.3, 0.31, 0.32, 0.3, 0.31],
                borderColor: '#00ff00',
                borderWidth: 2,
                fill: false,
                tension: 0.3
            }, {
                label: 'Ngưỡng nguy hiểm',
                data: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                borderColor: '#ff0000',
                borderWidth: 1,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 0.1, max: 0.4 },
                x: { display: false }
            },
            animation: false
        }
    });

    setInterval(() => {
        let newValue = (Math.random() * (0.35 - 0.25) + 0.25).toFixed(2);

        if (Math.random() < 0.1) {
            newValue = (Math.random() * (0.18 - 0.12) + 0.12).toFixed(2);
        }

        earChart.data.labels.push('');
        earChart.data.labels.shift();
        earChart.data.datasets[0].data.push(newValue);
        earChart.data.datasets[0].data.shift();

        if (newValue < 0.2) {
            earChart.data.datasets[0].borderColor = '#ff0000';
        } else {
            earChart.data.datasets[0].borderColor = '#00ff00';
        }

        earChart.update();
    }, 500);
}