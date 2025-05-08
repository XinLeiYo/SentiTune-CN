"""
SentiTune-CN - 情感分析核心
Created: 2025-05-08 13:06:20 UTC
Author: XinLeiYo
"""

class SentimentAnalyzer:
    def analyze(self, text: str) -> Dict[str, Any]:
        """分析文字情感"""
        try:
            s = SnowNLP(text)
            
            # 基礎情感分數
            sentiment_score = s.sentiments
            
            # 應用自定義字典進行調整
            words = s.words
            custom_scores = []
            word_weights = []
            
            for word in words:
                if word in self.custom_dict:
                    custom_scores.append(self.custom_dict[word])
                    # 根據詞語的極性增加權重
                    polarity = abs(self.custom_dict[word] - 0.5) * 2
                    word_weights.append(1 + polarity)
                else:
                    word_weights.append(1.0)
            
            # 使用加權平均計算調整後的分數
            if custom_scores:
                weighted_custom_score = sum(score * weight for score, weight 
                                         in zip(custom_scores, word_weights))
                total_weight = sum(word_weights)
                adjusted_score = (sentiment_score + weighted_custom_score / total_weight) / 2
            else:
                adjusted_score = sentiment_score
            
            # 改進的信心分數計算
            confidence_factors = [
                # 1. 基於情感極性的信心
                abs(adjusted_score - 0.5) * 2,
                
                # 2. 基於自定義字典匹配度的信心
                len(custom_scores) / len(words) if words else 0,
                
                # 3. 基於文本長度的信心 (較長文本可能更可靠)
                min(len(words) / 20, 1.0),  # 最多貢獻1.0的信心
                
                # 4. 基於情感一致性的信心
                1.0 if not custom_scores else 
                1.0 - abs(sentiment_score - sum(custom_scores) / len(custom_scores)) / 2
            ]
            
            # 計算加權平均的信心分數
            weights = [0.4, 0.3, 0.1, 0.2]  # 各因素的權重
            confidence_score = sum(factor * weight 
                                for factor, weight in zip(confidence_factors, weights))
            
            # 確保信心分數在合理範圍內
            confidence_score = max(0.3, min(confidence_score, 1.0))  # 設定最低信心為0.3
            
            # 使用優化後的閾值判斷情感類別
            if adjusted_score > self.thresholds["positive"]:
                sentiment = "正面"
            elif adjusted_score < self.thresholds["negative"]:
                sentiment = "負面"
            else:
                sentiment = "中性"
                # 對於中性評價，稍微降低信心分數
                confidence_score *= 0.9
            
            return {
                "狀態": "成功",
                "情感分數": adjusted_score,
                "情感類別": sentiment,
                "信心分數": confidence_score,
                "信心因素": {  # 新增：顯示各個信心因素的分數
                    "情感極性": confidence_factors[0],
                    "字典匹配": confidence_factors[1],
                    "文本長度": confidence_factors[2],
                    "情感一致性": confidence_factors[3]
                },
                "關鍵詞": s.keywords(3),
                "分析時間": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logging.error(f"分析過程發生錯誤: {str(e)}")
            return {
                "狀態": "失敗",
                "錯誤訊息": str(e)
            }