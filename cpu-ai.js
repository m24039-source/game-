// ==========================================
// 数理変換タクティクス：CPU（AI）思考エンジン
// ==========================================
import { runTransaction } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// ヘルパー関数: Firebaseのオブジェクト手札を配列に安全変換
function targetHandArray(handData) {
    if (!handData) return [];
    if (Array.isArray(handData)) return handData;
    return Object.values(handData);
}

// ターン交代処理（HTML側の共通処理をAI用にも実行）
function advanceTurn(currentData) {
    currentData.currentTurnIdx = (currentData.currentTurnIdx + 1) % currentData.turnOrder.length;
    currentData.absoluteTurnIdx++; 
    if (currentData.currentTurnIdx === 0) {
        currentData.turnCount++;
        const maxTurnTarget = currentData.config?.maxTurns || 10;
        if (currentData.turnCount > maxTurnTarget) {
            currentData.status = 'finished';
            currentData.log = (currentData.log || "") + `🏁【制限ターン終了】規定周回数を消化したためゲームを終了します。\n`;
            let highestScore = -1; let winnerName = "";
            Object.values(currentData.players).forEach(p => {
                if (p.score > highestScore) { highestScore = p.score; winnerName = p.name; }
            });
            currentData.log = (currentData.log || "") + `🏆 最終勝者は ${winnerName} (${highestScore} pt) です！\n`;
        }
    }
}

// タイマーリセット
function resetTurnTimerStock(currentData) {
    currentData.remainingTime = currentData.config?.turnLimitTime || 30;
    currentData.timerMode = "stock"; 
}

/**
 * 🤖 CPU（AI）の自動思考ロジック本体
 */
export function executeCPUTurn(roomRef, cpuId) {
    runTransaction(roomRef, (currentData) => {
        if (!currentData || currentData.status !== 'playing') return currentData;
        
        // 1. ターン開始ドローの自動処理 (1周目以外)
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

        let bestAction = { type: "pass", score: -1, cardIdx: -1, card1Idx: -1, card2Idx: -1, resVal: -1, opLabel: "" };

        // 2. 攻撃パターンの全探索 (1周目でなければ実行)
        if (!isFirstRound) {
            for (let i = 0; i < cpuHand.length; i++) {
                let attackNum = cpuHand[i];
                if (attackNum === 1) continue; // 1での攻撃は禁止

                let totalGained = 0;
                Object.keys(currentData.players).forEach(pid => {
                    if (pid === cpuId) return;
                    let targetHand = targetHandArray(currentData.players[pid].hand);
                    targetHand.forEach(cardNum => {
                        if (cardNum % attackNum === 0) totalGained += cardNum;
                    });
                });

                if (totalGained > 0 && totalGained > bestAction.score) {
                    bestAction = { type: "attack", score: totalGained, cardIdx: i, value: attackNum };
                }
            }
        }

        // 3. 攻撃手段がない場合、合成(足し算・引き算・操作A)を全探索して最も大きな数字が作れるペアを探す
        if (bestAction.type === "pass" && cpuHand.length >= 2) {
            for (let i = 0; i < cpuHand.length; i++) {
                for (let j = 0; j < cpuHand.length; j++) {
                    if (i === j) continue;
                    let n1 = cpuHand[i];
                    let n2 = cpuHand[j];

                    // 足し算
                    let addRes = n1 + n2;
                    if (addRes <= limit && addRes > bestAction.resVal) {
                        bestAction = { type: "op", card1Idx: i, card2Idx: j, resVal: addRes, opLabel: "➕ 足し算" };
                    }
                    // 引き算
                    let subRes = n1 - n2;
                    if (subRes > 0 && subRes > bestAction.resVal) {
                        bestAction = { type: "op", card1Idx: i, card2Idx: j, resVal: subRes, opLabel: "➖ 引き算" };
                    }
                    // 操作A (商×余)
                    if (n2 !== 0) {
                        let opARes = Math.floor(n1 / n2) * (n1 % n2);
                        if (opARes <= limit && opARes > bestAction.resVal) {
                            bestAction = { type: "op", card1Idx: i, card2Idx: j, resVal: opARes, opLabel: "🧩 操作A: 商×余" };
                        }
                    }
                }
            }
        }

        // 4. 割り出した最善手を実際に反映
        if (bestAction.type === "attack") {
            let attackNum = bestAction.value;
            Object.keys(currentData.players).forEach(pid => {
                if (pid === cpuId) return;
                let targetHand = targetHandArray(currentData.players[pid].hand);
                let kept = [];
                targetHand.forEach(cardNum => {
                    if (cardNum % attackNum === 0) {
                        currentData.log = (currentData.log || "") + ` 🎯 ${currentData.players[pid].name} の [${cardNum}] を撃破！\n`;
                    } else {
                        kept.push(cardNum);
                    }
                });
                currentData.players[pid].hand = kept;
            });
            cpuHand.splice(bestAction.cardIdx, 1);
            currentData.players[cpuId].hand = cpuHand;
            currentData.players[cpuId].score += bestAction.score;
            currentData.log = (currentData.log || "") + `⚔️ ${currentData.players[cpuId].name} が [${attackNum}] で最適攻撃を選択、 ${bestAction.score} 点強奪！\n`;
            
            const winTarget = currentData.config?.winScore || 100;
            if (currentData.players[cpuId].score >= winTarget) {
                currentData.status = 'finished';
                currentData.log = (currentData.log || "") + `🏆👑【決着】AIが目標点に達したため勝利しました！\n`;
            }
        } else if (bestAction.type === "op") {
            let v1 = cpuHand[bestAction.card1Idx];
            let v2 = cpuHand[bestAction.card2Idx];
            
            let idx1 = cpuHand.indexOf(v1); if(idx1 !== -1) cpuHand.splice(idx1, 1);
            let idx2 = cpuHand.indexOf(v2); if(idx2 !== -1) cpuHand.splice(idx2, 1);
            
            cpuHand.push(bestAction.resVal);
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + `🔮 ${currentData.players[cpuId].name} は自動演算を判断: ${bestAction.opLabel} で [${v1}] と [${v2}] ➡ [${bestAction.resVal}] を合成。\n`;
        } else {
            currentData.log = (currentData.log || "") + `💤 ${currentData.players[cpuId].name} は有効な手がなくパスしました。\n`;
        }

        // 手番終了、次へ
        advanceTurn(currentData);
        resetTurnTimerStock(currentData);
        return currentData;
    });
}
