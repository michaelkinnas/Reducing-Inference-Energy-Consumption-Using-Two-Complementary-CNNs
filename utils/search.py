def threshold_search(trues, preds_a, preds_b, preds_score_a, preds_score_b, param_range, reverse):
    acc = []
    acc_ps = []
    if not reverse:
        for threshold in param_range:            
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a >= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b > pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true
                        
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })
    else:
        for threshold in param_range:
            max_p_correct = 0
            max_p_correct_ps = 0
            max_p_model_b_usage = 0
            for true, pred_a, pred_b, pred_score_a, pred_score_b in zip(trues, preds_a, preds_b, preds_score_a, preds_score_b):
                if pred_score_a <= threshold:
                    max_p_correct += pred_a == true
                    max_p_correct_ps += pred_a == true
                else:
                    max_p_model_b_usage += 1
                    max_p_correct += pred_b == true
                    if pred_score_b < pred_score_a:
                        max_p_correct_ps += pred_b == true
                    else:
                        max_p_correct_ps += pred_a == true
        
            acc.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct,
                'model_b_usage': max_p_model_b_usage
            })

            acc_ps.append({
                'threshold' : threshold,
                'accuracy' : max_p_correct_ps,
                'model_b_usage': max_p_model_b_usage,
            })
    return (acc, acc_ps)