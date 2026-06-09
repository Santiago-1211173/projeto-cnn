// Charts Initialization

let chartPCA = null;
let chartNoiseCurve = null;

const Charts = {
    initPCAChart() {
        const ctx = document.getElementById('chart-pca').getContext('2d');
        
        chartPCA = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [] // Populated dynamically
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#e2e8f0' }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                return `Classe: ${ctx.dataset.label} | Reward: ${ctx.raw.reward}`;
                            }
                        }
                    }
                },
                animation: false // Too many points to animate smoothly
            }
        });
    },

    updatePCAChart(scatterData) {
        if (!chartPCA) return;
        
        // Group by action (class 0-9)
        const groups = {};
        for (let i = 0; i < 10; i++) groups[i] = [];
        
        scatterData.forEach(pt => {
            if(groups[pt.action]) {
                groups[pt.action].push(pt);
            }
        });
        
        // Colors palette (Neon / Vibrant)
        const colors = [
            '#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16',
            '#22c55e', '#10b981', '#06b6d4', '#3b82f6', '#8b5cf6'
        ];
        
        const datasets = [];
        for (let i = 0; i < 10; i++) {
            datasets.push({
                label: `Dígito ${i}`,
                data: groups[i],
                backgroundColor: colors[i] + '80', // Add transparency
                borderColor: colors[i],
                borderWidth: 1,
                pointRadius: 3
            });
        }
        
        chartPCA.data.datasets = datasets;
        chartPCA.update();
    },

    initNoiseCurveChart() {
        const canvas = document.getElementById('chart-noise-curve');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        chartNoiseCurve = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['0.0', '0.2', '0.4', '0.6', '0.8'],
                datasets: [
                    {
                        label: 'Hybrid System',
                        data: [],
                        borderColor: '#58A6FF',
                        backgroundColor: 'rgba(88, 166, 255, 0.1)',
                        borderWidth: 3,
                        pointBackgroundColor: '#58A6FF',
                        pointBorderColor: '#0D1117',
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 9,
                        fill: true,
                        tension: 0.3,
                        borderDash: []
                    },
                    {
                        label: 'RL-only',
                        data: [],
                        borderColor: '#7EE787',
                        backgroundColor: 'rgba(126, 231, 135, 0.05)',
                        borderWidth: 2.5,
                        pointBackgroundColor: '#7EE787',
                        pointBorderColor: '#0D1117',
                        pointBorderWidth: 2,
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        pointStyle: 'triangle',
                        fill: false,
                        tension: 0.3,
                        borderDash: [8, 4]
                    },
                    {
                        label: 'CNN-only',
                        data: [],
                        borderColor: '#F78166',
                        backgroundColor: 'rgba(247, 129, 102, 0.05)',
                        borderWidth: 2,
                        pointBackgroundColor: '#F78166',
                        pointBorderColor: '#0D1117',
                        pointBorderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 7,
                        pointStyle: 'rectRot',
                        fill: false,
                        tension: 0,
                        borderDash: [4, 4]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 800,
                    easing: 'easeOutQuart'
                },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Nível de Ruído',
                            color: '#94a3b8',
                            font: { size: 13, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: {
                            color: '#8B949E',
                            font: { size: 11 }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Accuracy (%)',
                            color: '#94a3b8',
                            font: { size: 13, weight: 'bold' }
                        },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: {
                            color: '#8B949E',
                            font: { size: 11 },
                            callback: function(value) { return value.toFixed(1) + '%'; }
                        },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#e2e8f0',
                            usePointStyle: true,
                            pointStyleWidth: 14,
                            padding: 16,
                            font: { size: 12, weight: '600' }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(13, 17, 23, 0.95)',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        titleColor: '#F0F6FC',
                        bodyColor: '#C9D1D9',
                        padding: 12,
                        callbacks: {
                            title: function(items) {
                                return `Ruído: ${items[0].label}`;
                            },
                            label: function(context) {
                                return ` ${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    },

    updateNoiseCurveChart(noiseData) {
        if (!chartNoiseCurve || !noiseData) return;

        chartNoiseCurve.data.labels = noiseData.noises.map(String);
        chartNoiseCurve.data.datasets[0].data = noiseData.hybrid;
        chartNoiseCurve.data.datasets[1].data = noiseData.rl;
        chartNoiseCurve.data.datasets[2].data = noiseData.cnn;

        chartNoiseCurve.update();
    }
};

window.Charts = Charts;
