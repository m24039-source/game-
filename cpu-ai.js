// ==========================================
// 数理変換タクティクス：CPU（AI）思考エンジン
// 【ディープラーニング（TensorFlow.js）強化学習・自己反省機能搭載版】
// ==========================================
import { runTransaction } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// AIの脳みそ（ニューラルネットワークモデル）
let aiNeuralModel = null;

// 🧠 AIの1試合分の記憶を溜めるメモリ空間
let gameMemory = [];

/**
 * 🧠 1. ディープラーニングの脳みそ（モデル）を初期化・作成
 */
function createAiModel() {
    if (aiNeuralModel) return aiNeuralModel;

    const model = tf.sequential();
    // 入力層 (盤面状態 30要素)
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [30] }));
    // 中間層
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    // 出力層 (0: パス・合成, 1: 攻撃, 2: 特殊合成) の3アクションの期待度
    model.add(tf.layers.dense({ units: 3, activation: 'linear' }));

    // 学習アルゴリズム（Adam）を設定
    model.compile({ optimizer: tf.train.adam(0.05), loss: 'meanSquaredError' });
    
    console.log("🤖 [DL-AI] 自己学習型ニューラルネットワークが起動しました。");
    return model;
}

/**
 * 📊 2. 盤面データをAIが理解できる数値の配列（30要素）に正規化
 */
function convertStateToVector(currentData, cpuId) {
    const vector = new Array(30).fill(0);
    if (!currentData || !currentData.players) return vector;

    const myHand = targetHandArray(currentData.players[cpuId]?.hand);
    for (let i = 0; i < Math.min(myHand.length, 10); i++) {
        vector[i] = myHand[i] / 150.0; // 150を最大値として 0.0〜1.0 に正規化
    }
    vector[10] = (currentData.players[cpuId]?.score || 0) / 100.0;

    const enemyId = Object.keys(currentData.players).find(pid => pid !== cpuId);
    if (enemyId) {
        const enemyHand = targetHandArray(currentData.players[enemyId]?.hand);
        for (let i = 0; i < Math.min(enemyHand.length, 10); i++) {
            vector[11 + i] = enemyHand[i] / 150.0;
        }
        vector[21] = (currentData.players[enemyId]?.score || 0) / 100.0;
    }
    vector[22] = (currentData.turnCount || 1) / 10.0;
    return vector;
}

/**
 * 🏋️‍♂️ 4. 【核心】試合の勝敗結果を受けて、脳みそを強化（学習）する関数
 */
async function trainAiModel(isWinner) {
    if (gameMemory.length === 0 || !aiNeuralModel) return;

    console.log(`🏋️‍♂️ [AI学習開始] この試合のAI行動回数: ${gameMemory.length}回 | 勝敗結果: ${isWinner ? "🏆 勝利！" : "❌ 敗北"}`);

    const states = [];
    const targets = [];

    // 記憶をひとつずつ取り出して「反省データ」を作る
    for (const memory of gameMemory) {
        states.push(memory.state);

        // 現在の脳の予測値（元々の考え）を一度取り出す
        let targetVector = [0, 0, 0];
        tf.tidy(() => {
            const pred = aiNeuralModel.predict(tf.tensor2d([memory.state]));
            targetVector = Array.from(pred.dataSync());
        });

        // 🧠 強化学習の報酬（リワード）ロジック
        // 勝った試合の行動だった場合は期待度をアゲる、負けた場合は下げる
        const reward = isWinner ? 2.0 : -1.5;
        targetVector[memory.action] += reward; 

        targets.push(targetVector);
    }

    // TensorFlow.js のテンソル形式に変換して一括学習（fit）を実行
    const xs = tf.tensor2d(states);
    const ys = tf.tensor2d(targets);

    // バックグラウンドでニューラルネットワークの重みを更新
    await aiNeuralModel.fit(xs, ys, {
        epochs: 5, // 5回反復して深く反省させる
        shuffle: true
    });

    // メモリの解放
    xs.dispose();
    ys.dispose();

    console.log("✨ [AI学習完了] 脳みそのアップデートが完了しました！次の試合から新戦術を試します。");
    gameMemory = []; // 次の試合のために記憶をリセット
}

// 共通ヘルパー関数
function targetHandArray(handData) {
    if (!handData) return [];
    if (Array.isArray(handData)) return handData;
    return Object.values(handData);
}

function advanceTurn(currentData, cpuId) {
    currentData.currentTurnIdx = (currentData.currentTurnIdx + 1) % currentData.turnOrder.length;
    currentData.absoluteTurnIdx++; 
    
    // もしゲームが終了（finished）になっていたら勝敗を判定して学習をトリガーする
    if (currentData.status === 'finished') {
        let highestScore = -1;
        let winnerId = "";
        Object.keys(currentData.players).forEach(pid => {
            if (currentData.players[pid].score > highestScore) {
                highestScore = currentData.players[pid].score;
                winnerId = pid;
            }
        });
        // 自分が勝者かどうかを判定して学習へ送る
        const isAiWinner = (winnerId === cpuId);
        setTimeout(() => { trainAiModel(isAiWinner); }, 100);
    }

    if (currentData.currentTurnIdx === 0) {
        currentData.turnCount++;
        const maxTurnTarget = currentData.config?.maxTurns || 10;
        if (currentData.turnCount > maxTurnTarget) {
            currentData.status = 'finished';
            currentData.log = (currentData.log || "") + `🏁【制限ターン終了】ゲームを終了します。\n`;
            let highestScore = -1; let winnerName = "";
            Object.values(currentData.players).forEach(p => {
                if (p.score > highestScore) { highestScore = p.score; winnerName = p.name; }
            });
            currentData.log = (currentData.log || "") + `🏆 最終勝者は ${winnerName} です！\n`;
            
            // ターン上限切れの場合の学習トリガー
            let winnerId = Object.keys(currentData.players).find(pid => currentData.players[pid].score === highestScore);
            setTimeout(() => { trainAiModel(winnerId === cpuId); }, 100);
        }
    }
}

function resetTurnTimerStock(currentData) {
    currentData.remainingTime = currentData.config?.turnLimitTime || 30;
    currentData.timerMode = "stock"; 
}

/**
 * 🤖 3. DL・強化学習対応型 CPU自動思考ロジック本体
 */
export function executeCPUTurn(roomRef, cpuId) {
    // 脳みその初期化
    aiNeuralModel = createAiModel();

    runTransaction(roomRef, (currentData) => {
        if (!currentData || currentData.status !== 'playing') return currentData;
        
        const activePlayerId = currentData.turnOrder ? currentData.turnOrder[currentData.currentTurnIdx] : null;
        if (activePlayerId !== cpuId) return currentData;

        // ドロー処理
        if (currentData.absoluteTurnIdx > 0) {
            const rule = currentData.config || { drawCount: 2, drawMaxNum: 20 };
            let cpuHand = targetHandArray(currentData.players[cpuId].hand);
            let drawnNums = [];
            for(let i = 0; i < rule.drawCount; i++) {
                let drawn = Math.floor(Math.random() * rule.drawMaxNum) + 1;
                cpuHand.push(drawn);
                drawnNums.push(`[${drawn}]`);
            }
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + `📥 AIドロー: ${currentData.players[cpuId].name} が ${drawnNums.join(', ')} を引きました。\n`;
        }

        let cpuHand = targetHandArray(currentData.players[cpuId].hand);
        let limit = currentData.config?.handLimitNum || 150;
        let isFirstRound = (currentData.turnCount === 1);

        // --- 🧠 脳みそ（TensorFlow.js）による推論と記憶 ---
        const stateVector = convertStateToVector(currentData, cpuId);
        let predictedActionCategory = 0; // 0: パス・通常合成, 1: 攻撃, 2: 特殊変換
        
        tf.tidy(() => {
            const inputTensor = tf.tensor2d([stateVector]);
            const prediction = aiNeuralModel.predict(inputTensor);
            predictedActionCategory = prediction.argMax(-1).dataSync()[0];
        });

        // 📝 今回の盤面状態と、脳が選んだアクションカテゴリを記憶メモリにストック
        gameMemory.push({
            state: stateVector,
            action: predictedActionCategory
        });

        // 最善手決定用変数
        let bestAction = { type: "pass", score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1, resVal2: -1, opLabel: "" };

        // カテゴリ1：攻撃
        if (predictedActionCategory === 1 && !isFirstRound) {
            for (let i = 0; i < cpuHand.length; i++) {
                let attackNum = cpuHand[i]; if (attackNum === 1) continue;
                let totalGained = 0;
                Object.keys(currentData.players).forEach(pid => {
                    if (pid === cpuId) return;
                    let targetHand = targetHandArray(currentData.players[pid].hand);
                    targetHand.forEach(cardNum => { if (cardNum % attackNum === 0) totalGained += cardNum; });
                });
                if (totalGained > 0 && totalGained > bestAction.score) {
                    bestAction = { type: "attack", score: totalGained, cardIdx: i, value: attackNum };
                }
            }
        }

        // カテゴリ2：高度な合成（操作BやCを優先）
        if (bestAction.type === "pass" && predictedActionCategory === 2) {
            // 操作C (最大公約数) を優先探索
            if (cpuHand.length >= 2) {
                for (let i = 0; i < cpuHand.length; i++) {
                    for (let j = 0; j < cpuHand.length; j++) {
                        if (i === j) continue;
                        let a = cpuHand[i], b = cpuHand[j];
                        while (b !== 0) { let t = b; b = a % b; a = t; }
                        if (a > 1 && (cpuHand[i] * a) <= limit) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: cpuHand[i] * a, opLabel: "🔮 操作C: 最大公約数掛け" };
                            break;
                        }
                    }
                    if (bestAction.type !== "pass") break;
                }
            }
        }

        // カテゴリ0 または他が不発の時のセーフティネット（通常合成）
        if (bestAction.type === "pass") {
            if (cpuHand.length >= 2) {
                for (let i = 0; i < cpuHand.length; i++) {
                    for (let j = 0; j < cpuHand.length; j++) {
                        if (i === j) continue;
                        let addRes = cpuHand[i] + cpuHand[j];
                        if (addRes <= limit && addRes > bestAction.resVal) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: addRes, opLabel: "➕ 足し算" };
                        }
                    }
                }
            }
        }

        // --- 実行の反映 ---
        if (bestAction.type === "attack") {
            let attackNum = bestAction.value;
            Object.keys(currentData.players).forEach(pid => {
                if (pid === cpuId) return;
                let targetHand = targetHandArray(currentData.players[pid].hand);
                let kept = [];
                targetHand.forEach(cardNum => {
                    if (cardNum % attackNum === 0) {
                        currentData.log = (currentData.log || "") + ` 🎯 ${currentData.players[pid].name} の [${cardNum}] を撃破！\n`;
                    } else { kept.push(cardNum); }
                });
                currentData.players[pid].hand = kept;
            });
            cpuHand.splice(bestAction.cardIdx, 1);
            currentData.players[cpuId].hand = cpuHand;
            currentData.players[cpuId].score += bestAction.score;
            currentData.log = (currentData.log || "") + `🧠 AI[DL選択]: [${attackNum}] で攻撃、 ${bestAction.score} 点強奪！\n`;
            
            const winTarget = currentData.config?.winScore || 100;
            if (currentData.players[cpuId].score >= winTarget) {
                currentData.status = 'finished';
                currentData.log = (currentData.log || "") + `🏆👑 AIが目標スコアに到達し、完全勝利しました！\n`;
            }
        } else if (bestAction.type === "op2") {
            let v1 = cpuHand[bestAction.card1Idx]; let v2 = cpuHand[bestAction.card2Idx];
            let idx1 = cpuHand.indexOf(v1); if(idx1 !== -1) cpuHand.splice(idx1, 1);
            let idx2 = cpuHand.indexOf(v2); if(idx2 !== -1) cpuHand.splice(idx2, 1);
            cpuHand.push(bestAction.resVal);
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + `🧠 AI[DL選択]: ${bestAction.opLabel} ([${v1}]&[${v2}] ➡ [${bestAction.resVal}])\n`;
        } else {
            currentData.log = (currentData.log || "") + `💤 AIは有効な手がなくパスしました。\n`;
        }

        // 手番交代
        advanceTurn(currentData, cpuId);
        resetTurnTimerStock(currentData);
        return currentData;
    });
}
