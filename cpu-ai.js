// ==========================================
// 📺 スマホ画面直結型：デバッグログ出力システム
// ==========================================
function logToScreen(message, isError = false) {
    let debugDiv = document.getElementById('ai-debug-log');
    if (!debugDiv) {
        // 画面の最下部にログ表示用の黒いボックスを強制的に作ります
        debugDiv = document.createElement('div');
        debugDiv.id = 'ai-debug-log';
        debugDiv.style.position = 'fixed';
        debugDiv.style.bottom = '0';
        debugDiv.style.left = '0';
        debugDiv.style.width = '100%';
        debugDiv.style.height = '150px';
        debugDiv.style.backgroundColor = 'rgba(0,0,0,0.85)';
        debugDiv.style.color = '#00ff00';
        debugDiv.style.fontFamily = 'monospace';
        debugDiv.style.fontSize = '12px';
        debugDiv.style.overflowY = 'scroll';
        debugDiv.style.padding = '10px';
        debugDiv.style.boxSizing = 'border-box';
        debugDiv.style.zIndex = '99999';
        debugDiv.style.pointerEvents = 'none'; // ゲームの邪魔をしない
        document.body.appendChild(debugDiv);
    }
    const line = document.createElement('div');
    if (isError) line.style.color = '#ff3333';
    line.innerText = `[${new Date().toLocaleTimeString()}] ${message}`;
    debugDiv.appendChild(line);
    debugDiv.scrollTop = debugDiv.scrollHeight;
}

// 画面全体の致命的エラーを捕まえて画面に表示
window.addEventListener('error', function(e) {
    logToScreen(`🚨 JSクラッシュ: ${e.message} (${e.filename}:${e.lineno})`, true);
});

// グローバル変数
let aiModel = null;
let isAIBrainLoading = false;

/**
 * 🧠 1. AIの脳みそロード関数
 * ※ゲーム開始を邪魔しないよう、完全に裏方で処理させます。
 */
async function loadAIBrain() {
    if (aiModel || isAIBrainLoading) return aiModel;
    isAIBrainLoading = true;
    
    const modelUrl = 'https://m24039-source.github.io/game-/tfjs_model/model.json'; 
    logToScreen("🧠 脳みそファイルのダウンロードを開始します...");

    try {
        const response = await fetch(modelUrl);
        if (!response.ok) throw new Error(`HTTPエラー! ステータス: ${response.status}`);
        const modelJson = await response.json();

        // Keras 3形式のノイズをクレンジング
        if (modelJson && modelJson.modelTopology && modelJson.modelTopology.model_config) {
            const layers = modelJson.modelTopology.model_config.config.layers || [];
            layers.forEach(layer => {
                if (layer.class_name === 'InputLayer' && layer.config && layer.config.batch_shape) {
                    layer.config.batch_input_shape = layer.config.batch_shape;
                }
                if (layer.config && layer.config.dtype && typeof layer.config.dtype === 'object') {
                    layer.config.dtype = 'float32';
                }
                if (layer.config) delete layer.config.quantization_config;
            });
            if (modelJson.modelTopology.model_config.config && typeof modelJson.modelTopology.model_config.config.dtype === 'object') {
                modelJson.modelTopology.model_config.config.dtype = 'float32';
            }
        }

        const baseUrl = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1);
        aiModel = await tf.loadLayersModel({
            load: async () => ({ modelTopology: modelJson.modelTopology, weightsManifest: modelJson.weightsManifest }),
            path: baseUrl
        });

        logToScreen("🎉 AI脳みそのロードに完全成功しました！");
        isAIBrainLoading = false;
        return aiModel;

    } catch (error) {
        logToScreen(`⚠️ 通常ロード失敗、擬似AI（Fallback）を起動します: ${error.message}`, true);
        try {
            const fallbackModel = tf.sequential();
            fallbackModel.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [30]}));
            fallbackModel.add(tf.layers.dense({units: 32, activation: 'relu'}));
            fallbackModel.add(tf.layers.dense({units: 3, activation: 'linear'}));
            fallbackModel.compile({optimizer: 'adam', loss: 'meanSquaredError'});
            aiModel = fallbackModel;
            logToScreen("🎯 擬似AIの起動に成功。ゲームプレイは可能です。");
            isAIBrainLoading = false;
            return aiModel;
        } catch (innerError) {
            logToScreen(`🚨 致命的: 擬似AIの作成すら失敗: ${innerError.message}`, true);
            isAIBrainLoading = false;
        }
    }
}

/**
 * 📊 2. 状態ベクトル変換
 */
function convertStateToVector(currentData, cpuId) {
    const vector = new Array(30).fill(0.0);
    if (!currentData || !currentData.players) return vector;

    const myPlayer = currentData.players[cpuId] || {};
    const myHand = myPlayer.hand || [];
    for (let i = 0; i < Math.min(myHand.length, 10); i++) {
        vector[i] = myHand[i] / 150.0;
    }
    vector[10] = (myPlayer.score || 0) / 100.0;

    const enemyId = Object.keys(currentData.players).find(pid => pid !== cpuId);
    if (enemyId) {
        const enemyPlayer = currentData.players[enemyId];
        const enemyHand = enemyPlayer.hand || [];
        for (let i = 0; i < Math.min(enemyHand.length, 10); i++) {
            vector[11 + i] = enemyHand[i] / 150.0;
        }
        vector[21] = (enemyPlayer.score || 0) / 100.0;
    }
    vector[22] = (currentData.turnCount || 1) / 10.0;
    return vector;
}

/**
 * 🤖 3. CPU思考メインロジック
 */
async function thinkCpuTurn(currentGameData, cpuId) {
    if (!currentGameData || currentGameData.status !== 'playing') return currentGameData;

    if (!aiModel) {
        logToScreen("🤖 AIが未ロードのため、急ぎでロードを試みます...");
        await loadAIBrain();
    }

    logToScreen(`🤖 CPU(${cpuId}) が思考中...`);
    const stateVector = convertStateToVector(currentGameData, cpuId);
    let actionCategory = 0;

    if (aiModel) {
        try {
            const inputTensor = tf.tensor2d([stateVector], [1, 30]);
            const prediction = aiModel.predict(inputTensor);
            const predictionData = await prediction.data();
            actionCategory = predictionData.indexOf(Math.max(...predictionData));
            
            inputTensor.dispose();
            prediction.dispose();
            logToScreen(`🧠 AI推論完了: カテゴリ [${actionCategory}] を選択。`);
        } catch (e) {
            logToScreen(`⚠️ 推論エラーのため通常行動を選択: ${e.message}`, true);
            actionCategory = 0;
        }
    }

    // 手札処理
    const cpuPlayer = currentGameData.players[cpuId];
    let cpuHand = [...(cpuPlayer.hand || [])];
    const limit = currentGameData.config?.handLimitNum || 150;
    const isFirstRound = (currentGameData.turnCount === 1);

    let bestAction = { type: 'pass', score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1 };

    // 攻撃判定
    if (actionCategory === 1 && !isFirstRound) {
        cpuHand.forEach((attackNum, i) => {
            if (attackNum === 1) return;
            let totalGained = 0;
            Object.keys(currentGameData.players).forEach(pid => {
                if (pid === cpuId) return;
                (currentGameData.players[pid].hand || []).forEach(cardNum => {
                    if (cardNum % attackNum === 0) totalGained += cardNum;
                });
            });
            if (totalGained > 0 && totalGained > bestAction.score) {
                bestAction = { type: 'attack', score: totalGained, cardIdx: i, value: attackNum };
            }
        });
    }

    // 特殊合成判定
    if (bestAction.type === 'pass' && actionCategory === 2) {
        if (cpuHand.length >= 2) {
            for (let i = 0; i < cpuHand.length; i++) {
                for (let j = 0; j < cpuHand.length; j++) {
                    if (i === j) continue;
                    let a = cpuHand[i], b = cpuHand[j];
                    while (b !== 0) { let temp = b; b = a % b; a = temp; }
                    if (a > 1 && (cpuHand[i] * a) <= limit) {
                        bestAction = { type: 'op2', card1Idx: i, card2Idx: j, resVal: cpuHand[i] * a };
                        break;
                    }
                }
                if (bestAction.type !== 'pass') break;
            }
        }
    }

    // 通常合成判定
    if (bestAction.type === 'pass') {
        if (cpuHand.length >= 2) {
            for (let i = 0; i < cpuHand.length; i++) {
                for (let j = 0; j < cpuHand.length; j++) {
                    if (i === j) continue;
                    const addRes = cpuHand[i] + cpuHand[j];
                    if (addRes <= limit && addRes > bestAction.resVal) {
                        bestAction = { type: 'op2', card1Idx: i, card2Idx: j, resVal: addRes };
                    }
                }
            }
        }
    }

    // ゲームデータへの反映
    if (bestAction.type === 'attack') {
        const attackNum = bestAction.value;
        logToScreen(`⚔️ CPUアクション: 攻撃 [倍数: ${attackNum}]`);
        Object.keys(currentGameData.players).forEach(pid => {
            if (pid === cpuId) return;
            currentGameData.players[pid].hand = currentGameData.players[pid].hand.filter(n => n % attackNum !== 0);
        });
        cpuHand.splice(bestAction.cardIdx, 1);
        currentGameData.players[cpuId].hand = cpuHand;
        currentGameData.players[cpuId].score = (currentGameData.players[cpuId].score || 0) + bestAction.score;
        if (currentGameData.players[cpuId].score >= (currentGameData.config?.winScore || 100)) currentGameData.status = 'finished';
    } else if (bestAction.type === 'op2') {
        logToScreen(`➕ CPUアクション: 合成 [結果: ${bestAction.resVal}]`);
        const popIndices = [bestAction.card1Idx, bestAction.card2Idx].sort((a, b) => b - a);
        popIndices.forEach(idx => cpuHand.splice(idx, 1));
        cpuHand.push(bestAction.resVal);
        currentGameData.players[cpuId].hand = cpuHand;
    } else {
        logToScreen(`💤 CPUアクション: パス`);
    }

    return currentGameData;
}

// 🚀 フリーズを絶対に防ぐため、読み込み完了後「5秒」待ってから非同期で静かにロードする
window.addEventListener('load', () => {
    setTimeout(() => {
        loadAIBrain().catch(e => logToScreen("ロードキャッチ: " + e.message, true));
    }, 5000);
});
