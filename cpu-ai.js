// ==========================================
// 数理変換タクティクス：CPU（AI）思考エンジン（中級・全操作対応版）
// ==========================================
import { runTransaction } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js";

// ヘルパー関数: Firebaseのオブジェクト手札を配列に安全変換
function targetHandArray(handData) {
    if (!handData) return [];
    if (Array.isArray(handData)) return handData;
    return Object.values(handData);
}

// ターン交代処理
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
 * 🤖 CPU（AI）の自動思考ロジック本体（中級）
 */
export function executeCPUTurn(roomRef, cpuId) {
    runTransaction(roomRef, (currentData) => {
        if (!currentData || currentData.status !== 'playing') return currentData;
        
        // 現在の手番がこのAIのものか再確認
        const activePlayerId = currentData.turnOrder ? currentData.turnOrder[currentData.currentTurnIdx] : null;
        if (activePlayerId !== cpuId) return currentData;

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

        // 最善手を記録するオブジェクト
        let bestAction = { 
            type: "pass", 
            score: -1, 
            cardIdx: -1, 
            card1Idx: -1, 
            card2Idx: -1, 
            resVal: -1, 
            resVal2: -1, // 操作Bの分裂用
            opLabel: "" 
        };

        // 2. 【最優先】攻撃パターンの全探索 (1周目でなければ実行)
        if (!isFirstRound) {
            for (let i = 0; i < cpuHand.length; i++) {
                let attackNum = cpuHand[i];
                if (attackNum === 1) continue; // 1での攻撃はルール禁止

                let totalGained = 0;
                Object.keys(currentData.players).forEach(pid => {
                    if (pid === cpuId) return;
                    let targetHand = targetHandArray(currentData.players[pid].hand);
                    targetHand.forEach(cardNum => {
                        if (cardNum % attackNum === 0) totalGained += cardNum;
                    });
                });

                // 最も多くのポイントを奪えるカードを攻撃用に選ぶ
                if (totalGained > 0 && totalGained > bestAction.score) {
                    bestAction = { type: "attack", score: totalGained, cardIdx: i, value: attackNum };
                }
            }
        }

        // 3. 攻撃手段がない場合、全ての変換・合成操作（A, B, C含む）を全探索
        if (bestAction.type === "pass") {
            
            // --- 2枚選ぶ合成（足し、引き、操作A、操作C）の探索 ---
            if (cpuHand.length >= 2) {
                for (let i = 0; i < cpuHand.length; i++) {
                    for (let j = 0; j < cpuHand.length; j++) {
                        if (i === j) continue;
                        let n1 = cpuHand[i];
                        let n2 = cpuHand[j];

                        // ① 足し算
                        let addRes = n1 + n2;
                        if (addRes <= limit && addRes > bestAction.resVal) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: addRes, opLabel: "➕ 足し算" };
                        }
                        // ② 引き算
                        let subRes = n1 - n2;
                        if (subRes > 0 && subRes > bestAction.resVal) {
                            bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: subRes, opLabel: "➖ 引き算" };
                        }
                        // ③ 操作A (商×余)
                        if (n2 !== 0) {
                            let opARes = Math.floor(n1 / n2) * (n1 % n2);
                            if (opARes <= limit && opARes > bestAction.resVal) {
                                bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: opARes, opLabel: "🧩 操作A: 商×余" };
                            }
                        }
                        // ④ 操作C (最大公約数掛け) ★新規追加
                        let a = n1, b = n2;
                        while (b !== 0) { let t = b; b = a % b; a = t; }
                        let gcd = a;
                        if (gcd > 1) {
                            let opCRes = n1 * gcd;
                            if (opCRes <= limit && opCRes > bestAction.resVal) {
                                bestAction = { type: "op2", card1Idx: i, card2Idx: j, resVal: opCRes, opLabel: "🔮 操作C: 最大公約数掛け" };
                            }
                        }
                    }
                }
            }

            // --- 1枚選ぶ合成（操作B: 桁の和で分裂）の探索 --- ★新規追加
            for (let i = 0; i < cpuHand.length; i++) {
                let num = cpuHand[i];
                let sum = String(num).split('').reduce((s, d) => s + parseInt(d), 0);
                if (sum > 0) {
                    let quotient = Math.floor(num / sum);
                    let remainder = num % sum;
                    
                    if (quotient <= limit && remainder <= limit) {
                        // AIの評価基準：分裂した結果、より大きな数字(商)が生まれるなら採用
                        if (quotient > bestAction.resVal) {
                            bestAction = { 
                                type: "op1", 
                                cardIdx: i, 
                                resVal: quotient, 
                                resVal2: remainder, 
                                origVal: num,
                                opLabel: "🔢 操作B: 桁和で分裂" 
                            };
                        }
                    }
                }
            }
        }

        // 4. 割り出した最善手を実際にFirebaseへ反映
        if (bestAction.type === "attack") {
            // 【攻撃の実行】
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
            
        } else if (bestAction.type === "op2") {
            // 【2枚合成の実行（足し・引き・A・C）】
            let v1 = cpuHand[bestAction.card1Idx];
            let v2 = cpuHand[bestAction.card2Idx];
            
            let idx1 = cpuHand.indexOf(v1); if(idx1 !== -1) cpuHand.splice(idx1, 1);
            let idx2 = cpuHand.indexOf(v2); if(idx2 !== -1) cpuHand.splice(idx2, 1);
            
            cpuHand.push(bestAction.resVal);
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + `🔮 ${currentData.players[cpuId].name} は自動演算を判断: ${bestAction.opLabel} で [${v1}] と [${v2}] ➡ [${bestAction.resVal}] を合成。\n`;
            
        } else if (bestAction.type === "op1") {
            // 【1枚合成の実行（操作B）】
            cpuHand.splice(bestAction.cardIdx, 1);
            cpuHand.push(bestAction.resVal);
            
            let logMsg = `🔮 ${currentData.players[cpuId].name} は自動演算を判断: ${bestAction.opLabel} により [${bestAction.origVal}] ➡ [${bestAction.resVal}]`;
            if (bestAction.resVal2 > 0) {
                cpuHand.push(bestAction.resVal2);
                logMsg += ` と [${bestAction.resVal2}] の2枚に分裂！\n`;
            } else {
                logMsg += ` に変換(余りなし)。\n`;
            }
            
            currentData.players[cpuId].hand = cpuHand;
            currentData.log = (currentData.log || "") + logMsg;
            
        } else {
            // 【パス】
            currentData.log = (currentData.log || "") + `💤 ${currentData.players[cpuId].name} は有効な手がなくパスしました。\n`;
        }

        // 手番を次のプレイヤーに進める
        advanceTurn(currentData);
        resetTurnTimerStock(currentData);
        return currentData;
    });
}
