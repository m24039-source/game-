// ==========================================
// 📺 スマホ画面直結型：デバッグログ出力システム
// ==========================================
function logToScreen(message, isError = false) {
    let debugDiv = document.getElementById('ai-debug-log');
    if (!debugDiv) {
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
        debugDiv.style.pointerEvents = 'none';
        document.body.appendChild(debugDiv);
    }
    const line = document.createElement('div');
    if (isError) line.style.color = '#ff3333';
    line.innerText = `[${new Date().toLocaleTimeString()}] ${message}`;
    debugDiv.appendChild(line);
    debugDiv.scrollTop = debugDiv.scrollHeight;
}

window.addEventListener('error', function(e) {
    logToScreen(`🚨 JSクラッシュ: ${e.message} (${e.filename}:${e.lineno})`, true);
});

let aiModel = null;
let isAIBrainLoading = false;
let cpuTurnInProgress = false;

/**
 * 🧠 1. AIの脳みそロード関数
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

    const cpuPlayer = currentGameData.players[cpuId];
    let cpuHand = [...(cpuPlayer.hand || [])];
    const limit = currentGameData.config?.handLimitNum || 150;
    const isFirstRound = (currentGameData.turnCount === 1);

    let bestAction = { type: 'pass', score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1 };

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

/**
 * ✅ グローバル関数：executeCPUTurn
 * 🔧 修正: グローバル関数として定義
 */
async function executeCPUTurn(roomRef, cpuId, runTransaction, cachedGameState) {
    if (cpuTurnInProgress) {
        logToScreen(`⏸️ スキップ: CPU既に思考中`);
        return;
    }
    cpuTurnInProgress = true;
    try {
        logToScreen(`🚀 executeCPUTurn 開始: ${cpuId}`);

        // Step1: トランザクションの外でAI推論（非同期処理はここで完結させる）
        if (!aiModel) {
            logToScreen("🤖 AIが未ロードのため急ぎロードします...");
            await loadAIBrain();
        }
        let actionCategory = 0;
        if (aiModel && cachedGameState) {
            try {
                const stateVector = convertStateToVector(cachedGameState, cpuId);
                const inputTensor = tf.tensor2d([stateVector], [1, 30]);
                const prediction = aiModel.predict(inputTensor);
                const predictionData = await prediction.data();
                actionCategory = predictionData.indexOf(Math.max(...predictionData));
                inputTensor.dispose();
                prediction.dispose();
                logToScreen(`🧠 AI推論完了: カテゴリ [${actionCategory}] を選択。`);
            } catch (e) {
                logToScreen(`⚠️ 推論エラー、パスします: ${e.message}`, true);
            }
        }

        // Step2: 同期トランザクションでゲーム状態を更新
        await runTransaction(roomRef, (currentData) => {
            if (!currentData || currentData.status !== 'playing') return currentData;

            const activeId = currentData.turnOrder[currentData.currentTurnIdx];
            if (activeId !== cpuId) {
                logToScreen(`⏸️ ${cpuId}のターンではないためスキップ`);
                return currentData;
            }

            const cpuPlayer = currentData.players[cpuId];
            if (!cpuPlayer) return currentData;

            let cpuHand = [...(Array.isArray(cpuPlayer.hand) ? cpuPlayer.hand : Object.values(cpuPlayer.hand || {}))];
            const limit = currentData.config?.handLimitNum || 150;
            const isFirstRound = (currentData.turnCount === 1);
            let bestAction = { type: 'pass', score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1 };

            if (actionCategory === 1 && !isFirstRound) {
                cpuHand.forEach((attackNum, i) => {
                    if (attackNum === 1) return;
                    let totalGained = 0;
                    Object.keys(currentData.players).forEach(pid => {
                        if (pid === cpuId) return;
                        (Array.isArray(currentData.players[pid].hand) ? currentData.players[pid].hand : Object.values(currentData.players[pid].hand || {})).forEach(n => {
                            if (n % attackNum === 0) totalGained += n;
                        });
                    });
                    if (totalGained > 0 && totalGained > bestAction.score) {
                        bestAction = { type: 'attack', score: totalGained, cardIdx: i, value: attackNum };
                    }
                });
            }

            if (bestAction.type === 'pass') {
                if (cpuHand.length >= 2) {
                    outer: for (let i = 0; i < cpuHand.length; i++) {
                        for (let j = 0; j < cpuHand.length; j++) {
                            if (i === j) continue;
                            const addRes = cpuHand[i] + cpuHand[j];
                            if (addRes <= limit) {
                                bestAction = { type: 'op2', card1Idx: i, card2Idx: j, resVal: addRes };
                                break outer;
                            }
                        }
                    }
                }
            }

            if (bestAction.type === 'attack') {
                const attackNum = bestAction.value;
                logToScreen(`⚔️ CPUアクション: 攻撃 [倍数: ${attackNum}]`);
                Object.keys(currentData.players).forEach(pid => {
                    if (pid === cpuId) return;
                    const h = Array.isArray(currentData.players[pid].hand) ? currentData.players[pid].hand : Object.values(currentData.players[pid].hand || {});
                    currentData.players[pid].hand = h.filter(n => n % attackNum !== 0);
                });
                cpuHand.splice(bestAction.cardIdx, 1);
                currentData.players[cpuId].hand = cpuHand;
                currentData.players[cpuId].score = (currentData.players[cpuId].score || 0) + bestAction.score;
                if (currentData.players[cpuId].score >= (currentData.config?.winScore || 100)) currentData.status = 'finished';
            } else if (bestAction.type === 'op2') {
                logToScreen(`➕ CPUアクション: 合成 [結果: ${bestAction.resVal}]`);
                [bestAction.card1Idx, bestAction.card2Idx].sort((a, b) => b - a).forEach(idx => cpuHand.splice(idx, 1));
                cpuHand.push(bestAction.resVal);
                currentData.players[cpuId].hand = cpuHand;
            } else {
                logToScreen(`💤 CPUアクション: パス`);
            }

            // ターン進行
            currentData.currentTurnIdx = (currentData.currentTurnIdx + 1) % currentData.turnOrder.length;
            currentData.absoluteTurnIdx++;
            if (currentData.currentTurnIdx === 0) {
                currentData.turnCount++;
                if (currentData.turnCount > (currentData.config?.maxTurns || 10)) currentData.status = 'finished';
            }

            return currentData;
        });

        logToScreen(`✅ CPU${cpuId} のターン処理完了！`);

    } catch (error) {
        logToScreen(`❌ executeCPUTurn でエラー: ${error.message}`, true);
    } finally {
        cpuTurnInProgress = false;
    }
}

/**
 * ✅ グローバル関数：loadBrain
 */
async function loadBrain() {
    try {
        logToScreen("🧠 ユーザーが脳みそロードをリクエスト");
        const result = await loadAIBrain();
        if (result) {
            logToScreen("✅ 脳みそロード完全成功！");
            return true;
        } else {
            logToScreen("⚠️ 脳みそロード失敗", true);
            return false;
        }
    } catch (error) {
        logToScreen(`❌ loadBrain エラー: ${error.message}`, true);
        return false;
    }
}

window.addEventListener('load', () => {
    setTimeout(() => {
        loadAIBrain().catch(e => logToScreen("ロードキャッチ: " + e.message, true));
    }, 5000);
});

export { executeCPUTurn, loadBrain };
