const API = {
    async getImages() {
        const res = await fetch('/api/images');
        return await res.json();
    },

    async predict(index, noiseLevel, threshold, k) {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                index: index,
                noise_level: noiseLevel,
                threshold: threshold,
                k: k
            })
        });
        return await res.json();
    },

    async getMemoryStats() {
        const res = await fetch('/api/memory_stats');
        return await res.json();
    },

    async getTrainingStatus() {
        const res = await fetch('/api/train/status');
        return await res.json();
    },

    async startTraining() {
        const res = await fetch('/api/train/start', { method: 'POST' });
        return await res.json();
    },

    async evaluateGlobal(dataset) {
        const res = await fetch('/api/evaluate/global', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset })
        });
        return await res.json();
    }
};

window.API = API;
