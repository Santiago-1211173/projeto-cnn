document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            const target = btn.getAttribute('data-target');
            document.getElementById(target).classList.add('active');
            
            // Load memory stats when tab is activated
            if (target === 'tab-memory') {
                loadMemoryStats();
            }
        });
    });

    // 3. Inference Logic
    let currentImages = [];
    let currentImageIndex = 0;
    
    const ui = {
        thresholdSlider: document.getElementById('threshold-slider'),
        thresholdVal: document.getElementById('threshold-val'),
        noiseSlider: document.getElementById('noise-slider'),
        noiseVal: document.getElementById('noise-val'),
        kSlider: document.getElementById('k-slider'),
        kVal: document.getElementById('k-val'),
        btnNext: document.getElementById('btn-next-image'),
        queryImage: document.getElementById('query-image'),
        trueLabelTxt: document.getElementById('true-label-txt'),
        
        nodeCnn: document.getElementById('node-cnn'),
        cnnPred: document.getElementById('cnn-pred'),
        cnnConf: document.getElementById('cnn-conf'),
        
        mDistTxt: document.getElementById('m-dist-txt'),
        mDistFill: document.getElementById('m-dist-fill'),
        mDistMarker: document.getElementById('m-dist-marker'),
        mDistThresh: document.getElementById('m-dist-thresh'),
        arrowIcon: document.getElementById('arrow-icon'),
        
        nodeKnn: document.getElementById('node-knn'),
        knnPred: document.getElementById('knn-pred'),
        
        finalPredTxt: document.getElementById('final-pred'),
        finalDecisionBox: document.getElementById('final-decision-box'),
        
        kCountDisplay: document.getElementById('k-count-display'),
        neighborsGrid: document.getElementById('neighbors-grid')
    };

    // Update Slider Labels
    ui.thresholdSlider.addEventListener('input', (e) => ui.thresholdVal.textContent = e.target.value);
    ui.noiseSlider.addEventListener('input', (e) => ui.noiseVal.textContent = e.target.value);
    ui.kSlider.addEventListener('input', (e) => ui.kVal.textContent = e.target.value);

    // Fetch initial images
    async function loadTestImages() {
        try {
            currentImages = await API.getImages();
            currentImageIndex = 0;
            runPrediction();
        } catch (e) {
            console.error("Failed to load images", e);
        }
    }

    async function runPrediction() {
        if (currentImages.length === 0) return;
        
        const imgObj = currentImages[currentImageIndex];
        const noise = parseFloat(ui.noiseSlider.value);
        const threshold = parseFloat(ui.thresholdSlider.value);
        const k = parseInt(ui.kSlider.value);
        
        // Show loading state
        ui.nodeCnn.classList.remove('active-route');
        ui.nodeKnn.classList.remove('active-route');
        ui.nodeKnn.classList.add('inactive');
        ui.arrowIcon.className = 'arrow-icon';
        ui.arrowIcon.textContent = '⟷';
        ui.mDistFill.style.width = '0%';
        ui.kCountDisplay.textContent = k;
        ui.neighborsGrid.innerHTML = '<div class="neighbor-placeholder">Inferindo...</div>';
        
        try {
            const res = await API.predict(imgObj.index, noise, threshold, k);
            
            // Update Query Panel
            ui.queryImage.src = `data:image/png;base64,${res.query_image_base64}`;
            ui.trueLabelTxt.textContent = res.true_label;
            
            // Update CNN Node
            ui.cnnPred.textContent = res.cnn_pred;
            ui.cnnConf.textContent = (res.cnn_conf * 100).toFixed(1);
            
            // Update Arbitrator Center
            ui.mDistTxt.textContent = res.m_dist.toFixed(2);
            ui.mDistThresh.textContent = threshold.toFixed(1);
            
            // Calculate Gauge Percentages (Max visual scale is usually around threshold * 2)
            const maxScale = Math.max(threshold * 1.5, res.m_dist * 1.2, 30);
            let fillPct = Math.min((res.m_dist / maxScale) * 100, 100);
            let markerPct = Math.min((threshold / maxScale) * 100, 100);
            
            ui.mDistFill.style.width = `${fillPct}%`;
            ui.mDistMarker.style.left = `${markerPct}%`;
            
            // Update k-NN Node
            ui.knnPred.textContent = res.knn_pred === -1 ? '-' : res.knn_pred;
            
            if (res.routed_to === 'CNN') {
                ui.nodeCnn.classList.add('active-route');
                ui.finalPredTxt.textContent = res.cnn_pred;
                
                ui.mDistFill.style.background = 'var(--accent-blue)';
                ui.arrowIcon.className = 'arrow-icon arrow-left';
                ui.arrowIcon.textContent = '⟵';
                
                ui.neighborsGrid.innerHTML = '<div class="neighbor-placeholder">CNN Confiante. k-NN não invocado.</div>';
            } else {
                ui.nodeKnn.classList.remove('inactive');
                ui.nodeKnn.classList.add('active-route');
                ui.finalPredTxt.textContent = res.knn_pred;
                
                ui.mDistFill.style.background = 'var(--accent-gold)';
                ui.arrowIcon.className = 'arrow-icon arrow-right';
                ui.arrowIcon.textContent = '⟶';
                
                // Render Neighbors
                renderNeighbors(res.nearest_neighbors);
            }
            
            // Final decision box style
            if (res.final_pred === res.true_label) {
                ui.finalDecisionBox.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.4)';
                ui.finalDecisionBox.style.borderColor = 'var(--accent-green)';
            } else {
                ui.finalDecisionBox.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.4)';
                ui.finalDecisionBox.style.borderColor = 'var(--accent-red)';
            }
            
            // Update Rewards Divergent Bars
            renderRewards(res.expected_rewards);
            
        } catch(e) {
            console.error("Prediction failed", e);
        }
    }

    function renderNeighbors(neighbors) {
        ui.neighborsGrid.innerHTML = '';
        neighbors.forEach(n => {
            const card = document.createElement('div');
            card.className = 'neighbor-card';
            
            const rClass = n.is_positive ? 'pos' : 'neg';
            const rText = n.reward > 0 ? '+1' : '-1';
            
            card.innerHTML = `
                <img src="data:image/png;base64,${n.base64}" alt="Neighbor">
                <div class="n-info-row">
                    <span class="n-tag" title="Label Real (Ground Truth)">Real: <strong>${n.true_label}</strong></span>
                    <span class="n-tag" title="Ação do RL">Ação: <strong>${n.action}</strong></span>
                </div>
                <div class="n-reward ${rClass}">Reward: ${rText}</div>
                <div class="n-dist">Dist L2: <strong>${n.distance.toFixed(3)}</strong></div>
            `;
            ui.neighborsGrid.appendChild(card);
        });
    }

    function renderRewards(rewards) {
        const list = document.getElementById('rewards-list');
        if (!list) return;
        list.innerHTML = '';
        
        // Find max absolute value for scaling, but never scale below 1.0
        const maxAbs = Math.max(...rewards.map(Math.abs), 1);
        const maxVal = Math.max(...rewards);
        
        rewards.forEach((val, idx) => {
            const item = document.createElement('div');
            item.className = 'reward-item';
            
            const widthPct = (Math.abs(val) / maxAbs) * 100;
            let leftFill = '';
            let rightFill = '';
            
            if (val < 0) {
                leftFill = `<div class="reward-bar-fill" style="width: ${widthPct}%;"><span class="reward-value">${val.toFixed(2)}</span></div>`;
            } else if (val > 0) {
                const winnerClass = val === maxVal ? 'fill-winner' : '';
                rightFill = `<div class="reward-bar-fill ${winnerClass}" style="width: ${widthPct}%;"><span class="reward-value">+${val.toFixed(2)}</span></div>`;
            }
            
            item.innerHTML = `
                <span class="reward-label">Classe ${idx}</span>
                <div class="reward-bar-bg">
                    <div class="reward-half left">${leftFill}</div>
                    <div class="reward-half right">${rightFill}</div>
                </div>
            `;
            list.appendChild(item);
        });
    }

    // Handlers
    ui.btnNext.addEventListener('click', () => {
        currentImageIndex = (currentImageIndex + 1) % currentImages.length;
        runPrediction();
    });
    
    // Re-run on slider changes
    ui.thresholdSlider.addEventListener('change', runPrediction);
    ui.noiseSlider.addEventListener('change', runPrediction);
    ui.kSlider.addEventListener('change', runPrediction);


    // 4. Memory Detalhe Tab Logic
    const btnRefreshMemory = document.getElementById('btn-refresh-memory');
    
    async function loadMemoryStats() {
        try {
            const stats = await API.getMemoryStats();
            document.getElementById('mem-size').textContent = stats.size.toLocaleString();
            document.getElementById('mem-pos-pct').textContent = stats.reward_positive_pct ? stats.reward_positive_pct.toFixed(1) + '%' : '-';
            document.getElementById('mem-mean-reward').textContent = stats.reward_mean ? stats.reward_mean.toFixed(2) : '-';
            
            if(stats.memory_sample) {
                const listBody = document.getElementById('memory-list-body');
                listBody.innerHTML = '';
                
                stats.memory_sample.forEach(mem => {
                    const row = document.createElement('div');
                    row.className = 'memory-row';
                    
                    // Generate state HTML (now showing original image)
                    let stateHtml = '<span class="state-val" style="color:var(--text-muted)">Sem Imagem</span>';
                    if (mem.base64) {
                        stateHtml = `
                            <img src="data:image/png;base64,${mem.base64}" alt="Original" style="width: 30px; height: 30px; border-radius: 4px; border: 1px solid var(--border-glass); image-rendering: pixelated; margin-right: 8px;">
                            <span class="n-tag" style="flex: 0 0 auto;" title="Label Real (Ground Truth)">Real: <strong>${mem.true_label}</strong></span>
                        `;
                    }
                    
                    const rewardClass = mem.reward > 0 ? 'pos' : 'neg';
                    const rewardText = mem.reward > 0 ? `+${mem.reward}` : `${mem.reward}`;
                    
                    row.innerHTML = `
                        <div class="mem-id">#${mem.index}</div>
                        <div class="mem-state" style="padding-right: 0; align-items: center;">${stateHtml}</div>
                        <div class="mem-action">${mem.action}</div>
                        <div class="mem-reward ${rewardClass}">${rewardText}</div>
                    `;
                    listBody.appendChild(row);
                });
            }
        } catch(e) {
            console.error(e);
        }
    }
    
    btnRefreshMemory.addEventListener('click', loadMemoryStats);


    // 5. Training / Sementeira Tab Logic
    const btnStartTraining = document.getElementById('btn-start-training');
    const termOutput = document.getElementById('terminal-output');
    const progFill = document.getElementById('training-progress-fill');
    const progTxt = document.getElementById('training-progress-txt');
    const trainCnnAcc = document.getElementById('train-cnn-acc');
    const trainMemSize = document.getElementById('train-mem-size');
    const agentIndicator = document.getElementById('agent-status-indicator');
    
    let pollInterval = null;

    function appendTermLine(txt) {
        const div = document.createElement('div');
        div.className = 'term-line';
        div.textContent = txt;
        termOutput.appendChild(div);
        termOutput.scrollTop = termOutput.scrollHeight;
    }

    async function pollTrainingStatus() {
        try {
            const status = await API.getTrainingStatus();
            
            // Update progress
            progFill.style.width = status.progress + '%';
            progTxt.textContent = status.progress.toFixed(1) + '%';
            
            if (status.cnn_acc > 0) trainCnnAcc.textContent = status.cnn_acc.toFixed(1) + '%';
            if (status.memory_size > 0) trainMemSize.textContent = status.memory_size.toLocaleString();
            
            // Update terminal
            termOutput.innerHTML = '';
            status.logs.forEach(appendTermLine);
            
            if (!status.is_running && status.progress >= 100) {
                clearInterval(pollInterval);
                btnStartTraining.disabled = false;
                agentIndicator.innerHTML = '<div class="dot active"></div> Agent Ready (Updated)';
                
                // Force a reload of the memory mapping stats if we go there
                loadMemoryStats();
            }
        } catch(e) {
            console.error("Polling failed", e);
        }
    }

    btnStartTraining.addEventListener('click', async () => {
        btnStartTraining.disabled = true;
        termOutput.innerHTML = '';
        appendTermLine("> Inicializando Sementeira Progressiva 128D...");
        agentIndicator.innerHTML = '<div class="dot" style="background:#f59e0b; box-shadow: 0 0 8px #f59e0b;"></div> Agent Training...';
        
        try {
            await API.startTraining();
            pollInterval = setInterval(pollTrainingStatus, 500);
        } catch(e) {
            console.error(e);
            btnStartTraining.disabled = false;
        }
    });

    // 6. Evaluation Tab Logic
    const btnRunEval = document.getElementById('btn-run-evaluation');
    const evalSamplesGrid = document.getElementById('evaluation-samples-grid');

    if (btnRunEval) {
        btnRunEval.addEventListener('click', async () => {
            const dataset = document.querySelector('input[name="eval-dataset"]:checked').value;
            btnRunEval.disabled = true;
            btnRunEval.textContent = '⏳ A AVALIAR...';
            evalSamplesGrid.innerHTML = '<div class="neighbor-placeholder">A processar avaliação global...</div>';

            // Reset KPIs
            ['eval-hybrid-acc', 'eval-cnn-acc', 'eval-rl-acc', 'eval-rl-rate'].forEach(id => {
                document.getElementById(id).textContent = '...';
            });

            try {
                const res = await API.evaluateGlobal(dataset);

                // Update KPI cards
                document.getElementById('eval-hybrid-acc').textContent = res.global.hybrid_acc.toFixed(2);
                document.getElementById('eval-cnn-acc').textContent = res.global.cnn_acc.toFixed(2);
                document.getElementById('eval-rl-acc').textContent = res.global.rl_acc.toFixed(2);
                document.getElementById('eval-rl-rate').textContent = res.global.rl_rate.toFixed(2);

                // Update noise curve chart
                Charts.updateNoiseCurveChart(res.noise_curve);

                // Render OOD routed samples
                evalSamplesGrid.innerHTML = '';
                if (res.routed_samples && res.routed_samples.length > 0) {
                    res.routed_samples.forEach(sample => {
                        const card = document.createElement('div');
                        card.className = 'neighbor-card';

                        const isCorrect = sample.knn_action === sample.true_label;
                        const borderColor = isCorrect ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.5)';

                        card.style.borderColor = borderColor;
                        card.innerHTML = `
                            <img src="data:image/png;base64,${sample.base64}" alt="OOD Sample">
                            <div class="n-info-row">
                                <span class="n-tag" title="Label Real">Real: <strong>${sample.true_label}</strong></span>
                                <span class="n-tag" title="CNN Pred">CNN: <strong>${sample.cnn_pred}</strong></span>
                            </div>
                            <div class="n-info-row">
                                <span class="n-tag" title="k-NN Action">RL: <strong>${sample.knn_action}</strong></span>
                                <span class="n-tag" title="Mahalanobis">d<sub>M</sub>: <strong>${sample.distance.toFixed(1)}</strong></span>
                            </div>
                        `;
                        evalSamplesGrid.appendChild(card);
                    });
                } else {
                    evalSamplesGrid.innerHTML = '<div class="neighbor-placeholder">Nenhuma amostra encaminhada ao RL.</div>';
                }
            } catch (e) {
                console.error('Evaluation failed', e);
                evalSamplesGrid.innerHTML = '<div class="neighbor-placeholder" style="color:var(--accent-red)">Erro na avaliação. Verifique a consola.</div>';
            } finally {
                btnRunEval.disabled = false;
                btnRunEval.textContent = '▶ INICIAR AVALIAÇÃO GLOBAL';
            }
        });
    }

    // Boot
    Charts.initNoiseCurveChart();
    loadTestImages();
});
