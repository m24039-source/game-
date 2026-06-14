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
            // フォールバック: Embedding+MaxPool構造のダミーモデル（7クラス）
            const myIn    = tf.input({shape: [MAX_HAND], dtype: 'int32', name: 'my_hand'});
            const enmIn   = tf.input({shape: [MAX_HAND], dtype: 'int32', name: 'enemy_hand'});
            const scIn    = tf.input({shape: [8],        name: 'scalars'});
            const emb     = tf.layers.embedding({inputDim: 151, outputDim: 16, maskZero: true, name: 'card_emb'});
            const myPool  = tf.layers.globalMaxPooling1d({name: 'my_pool'}).apply(emb.apply(myIn));
            const enmPool = tf.layers.globalMaxPooling1d({name: 'enemy_pool'}).apply(emb.apply(enmIn));
            const concat  = tf.layers.concatenate({name: 'combined'}).apply([myPool, enmPool, scIn]);
            const d1      = tf.layers.dense({units: 64, activation: 'relu'}).apply(concat);
            const out     = tf.layers.dense({units: 7, activation: 'softmax', name: 'action'}).apply(d1);
            const fallbackModel = tf.model({inputs: [myIn, enmIn, scIn], outputs: out});
            fallbackModel.compile({optimizer: 'adam', loss: 'categoricalCrossentropy'});
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

const MAX_HAND = 20;

/**
 * 📊 2. モデル入力変換（Embedding対応・可変手札対応）
 */
function convertStateToInputs(currentData, cpuId) {
    if (!currentData || !currentData.players) return null;
    const myPlayer  = currentData.players[cpuId] || {};
    const myHand    = Array.isArray(myPlayer.hand) ? myPlayer.hand : Object.values(myPlayer.hand || {});
    const enemyId   = Object.keys(currentData.players).find(p => p !== cpuId);
    const enemyHand = enemyId
        ? (Array.isArray(currentData.players[enemyId].hand)
            ? currentData.players[enemyId].hand
            : Object.values(currentData.players[enemyId].hand || {}))
        : [];

    // 0パディング（最大MAX_HAND枚、0=空きスロット）
    const pad = (arr) => {
        const sliced = arr.slice(0, MAX_HAND);
        return [...sliced, ...new Array(MAX_HAND - sliced.length).fill(0)];
    };

    const cfg = currentData.config || {};
    const scalars = [
        (myPlayer.score || 0) / 100.0,
        (enemyId ? (currentData.players[enemyId].score || 0) : 0) / 100.0,
        (currentData.turnCount || 1) / 10.0,
        myHand.length   / MAX_HAND,
        enemyHand.length / MAX_HAND,
        (cfg.winScore       || 100) / 200.0,
        (cfg.maxTurns       || 10)  / 20.0,
        (cfg.initHandCount  || 5)   / 10.0,
    ];

    return { myPad: pad(myHand), enemyPad: pad(enemyHand), scalars };
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
                const inputs = convertStateToInputs(cachedGameState, cpuId);
                if (inputs) {
                    const myTensor    = tf.tensor2d([inputs.myPad],    [1, MAX_HAND], 'int32');
                    const enemyTensor = tf.tensor2d([inputs.enemyPad], [1, MAX_HAND], 'int32');
                    const scTensor    = tf.tensor2d([inputs.scalars],  [1, 8]);
                    const prediction  = aiModel.predict({
                        'my_hand':    myTensor,
                        'enemy_hand': enemyTensor,
                        'scalars':    scTensor,
                    });
                    const predictionData = await prediction.data();
                    actionCategory = predictionData.indexOf(Math.max(...predictionData));
                    myTensor.dispose(); enemyTensor.dispose(); scTensor.dispose(); prediction.dispose();
                    logToScreen(`🧠 AI推論完了: カテゴリ [${actionCategory}] を選択。`);
                }
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

            // ─── ヘルパー関数（同期） ───
            const cpuGcd = (a, b) => { while(b){let t=b;b=a%b;a=t;} return a; };
            const getEnemyHand = () => {
                const eid = Object.keys(currentData.players).find(p => p !== cpuId);
                return eid ? (Array.isArray(currentData.players[eid].hand) ? currentData.players[eid].hand : Object.values(currentData.players[eid].hand || {})) : [];
            };

            // ─── 各操作の最善候補を探す ───
            let bestAttack = null, bestAttackGain = 0;
            let bestAdd = null, bestAddVal = -1;
            let bestSub = null, bestSubVal = 0;
            let bestDp = null, bestDpVal = -1;
            let bestDsd = null, bestDsdVal = -1;
            let bestGm = null, bestGmVal = -1;

            // 攻撃候補
            if (!isFirstRound) {
                const enemyH = getEnemyHand();
                cpuHand.forEach((atk, i) => {
                    if (atk === 1) return;
                    const gain = enemyH.reduce((s, n) => n % atk === 0 ? s + n : s, 0);
                    if (gain > bestAttackGain) { bestAttackGain = gain; bestAttack = { idx: i, value: atk, gain }; }
                });
            }

            // 2枚操作候補
            if (cpuHand.length >= 2) {
                for (let i = 0; i < cpuHand.length; i++) {
                    for (let j = 0; j < cpuHand.length; j++) {
                        if (i === j) continue;
                        const a = cpuHand[i], b = cpuHand[j];
                        // 足し算
                        const addR = a + b;
                        if (addR <= limit && addR > bestAddVal) { bestAddVal = addR; bestAdd = {i, j, res: addR}; }
                        // 引き算
                        const subR = a - b;
                        if (subR > 0 && subR > bestSubVal) { bestSubVal = subR; bestSub = {i, j, res: subR}; }
                        // 商×余
                        if (b !== 0) {
                            const dpR = Math.floor(a/b) * (a%b);
                            if (dpR > 0 && dpR <= limit && dpR > bestDpVal) { bestDpVal = dpR; bestDp = {i, j, res: dpR}; }
                        }
                        // GCD掛け算
                        const g = cpuGcd(a, b);
                        if (g > 1) {
                            const gmR = a * g;
                            if (gmR <= limit && gmR > bestGmVal) { bestGmVal = gmR; bestGm = {i, j, res: gmR}; }
                        }
                    }
                }
            }

            // 桁和で分裂候補
            cpuHand.forEach((num, i) => {
                const ds = String(num).split('').reduce((s,d) => s+parseInt(d), 0);
                if (ds === 0) return;
                const q = Math.floor(num / ds);
                if (q <= limit && q > bestDsdVal) { bestDsdVal = q; bestDsd = {idx: i, num, q, r: num % ds}; }
            });

            // ─── AIカテゴリに従って行動選択（フォールバックあり） ───
            let chosen = null;
            const myScore = currentData.players[cpuId].score || 0;
            const winScore = currentData.config?.winScore || 100;

            // カテゴリ1=攻撃, 2=足し算, 3=引き算, 4=商×余, 5=桁和, 6=GCD
            if (actionCategory === 1 && bestAttack) chosen = 'attack';
            else if (actionCategory === 2 && bestAdd) chosen = 'add';
            else if (actionCategory === 3 && bestSub) chosen = 'sub';
            else if (actionCategory === 4 && bestDp)  chosen = 'dp';
            else if (actionCategory === 5 && bestDsd) chosen = 'dsd';
            else if (actionCategory === 6 && bestGm)  chosen = 'gm';

            // フォールバック: AIが選んだ操作できないなら他の操作
            if (!chosen) {
                if (bestAttack && myScore + bestAttack.gain >= winScore) chosen = 'attack';
                else if (bestGm && bestGmVal >= 30)  chosen = 'gm';
                else if (bestAttack && bestAttackGain >= 10) chosen = 'attack';
                else if (bestAdd && bestAddVal >= 20) chosen = 'add';
                else if (bestDp  && bestDpVal  >= 15) chosen = 'dp';
                else if (bestAttack) chosen = 'attack';
                else if (bestGm)  chosen = 'gm';
                else if (bestDsd) chosen = 'dsd';
                else if (bestAdd) chosen = 'add';
                else if (bestDp)  chosen = 'dp';
                else if (bestSub) chosen = 'sub';
            }

            // ─── 行動実行 ───
            if (chosen === 'attack') {
                const { idx, value, gain } = bestAttack;
                logToScreen(`⚔️ CPU: 攻撃 [${value}] → ${gain}点獲得`);
                const eid = Object.keys(currentData.players).find(p => p !== cpuId);
                if (eid) {
                    const eh = Array.isArray(currentData.players[eid].hand) ? currentData.players[eid].hand : Object.values(currentData.players[eid].hand || {});
                    currentData.players[eid].hand = eh.filter(n => n % value !== 0);
                }
                cpuHand.splice(idx, 1);
                currentData.players[cpuId].score = (currentData.players[cpuId].score || 0) + gain;
                currentData.log = (currentData.log||'') + `⚔️ CPU が [${value}] で攻撃！ ${gain}点獲得\n`;
                if (currentData.players[cpuId].score >= winScore) currentData.status = 'finished';
            } else if (chosen === 'add') {
                const {i,j,res} = bestAdd;
                logToScreen(`➕ CPU: 足し算 [${cpuHand[i]}+${cpuHand[j]}=${res}]`);
                cpuHand = cpuHand.filter((_,k)=>k!==i&&k!==j); cpuHand.push(res);
                currentData.log = (currentData.log||'') + `➕ CPU: 足し算 → [${res}]\n`;
            } else if (chosen === 'sub') {
                const {i,j,res} = bestSub;
                logToScreen(`➖ CPU: 引き算 [${cpuHand[i]}-${cpuHand[j]}=${res}]`);
                cpuHand = cpuHand.filter((_,k)=>k!==i&&k!==j); cpuHand.push(res);
                currentData.log = (currentData.log||'') + `➖ CPU: 引き算 → [${res}]\n`;
            } else if (chosen === 'dp') {
                const {i,j,res} = bestDp;
                logToScreen(`🧩 CPU: 商×余 [${cpuHand[i]}÷${cpuHand[j]}=${res}]`);
                cpuHand = cpuHand.filter((_,k)=>k!==i&&k!==j); cpuHand.push(res);
                currentData.log = (currentData.log||'') + `🧩 CPU: 商×余 → [${res}]\n`;
            } else if (chosen === 'dsd') {
                const {idx,num,q,r} = bestDsd;
                logToScreen(`🔢 CPU: 桁和分裂 [${num}→${q}${r>0?'+'+r:''}]`);
                cpuHand.splice(idx, 1); cpuHand.push(q); if (r > 0) cpuHand.push(r);
                currentData.log = (currentData.log||'') + `🔢 CPU: 桁和分裂 [${num}] → [${q}]${r>0?` と [${r}]`:''}\n`;
            } else if (chosen === 'gm') {
                const {i,j,res} = bestGm;
                const g = cpuGcd(cpuHand[i], cpuHand[j]);
                logToScreen(`🔮 CPU: GCD掛け [${cpuHand[i]}×GCD(${cpuHand[i]},${cpuHand[j]})=${res}]`);
                cpuHand = cpuHand.filter((_,k)=>k!==i&&k!==j); cpuHand.push(res);
                currentData.log = (currentData.log||'') + `🔮 CPU: GCD掛け → [${res}]\n`;
            } else {
                logToScreen(`💤 CPU: パス`);
                currentData.log = (currentData.log||'') + `💤 CPU: パス\n`;
            }
            currentData.players[cpuId].hand = cpuHand;

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
