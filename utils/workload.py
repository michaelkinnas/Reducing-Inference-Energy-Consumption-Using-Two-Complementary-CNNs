from torch import inference_mode, argmax
from utils.monitor import ResourceMonitor
from tqdm.auto import tqdm


def single(model, valset, device):
    response_times = []
    preds = []
    trues = []
    
    monitor = ResourceMonitor()

    from torch import softmax

    input('RESET POWER METER AND PRESS ENTER...')

    with inference_mode():
        for _, (X, y) in tqdm(enumerate(valset), total=len(valset)):

            X = X.unsqueeze(dim=0).to(device)

            monitor.record_cpu_util()
            
            logits = model(X)
            probs = softmax(logits, dim=1)
            pred_label = argmax(probs, dim=1).item()

            monitor.record_cpu_util()

            response_times.append({
                'cpu_util': monitor.get_last_cpu_util(),
                'timestamp': monitor.get_last_timestamp(),
                'response_time': monitor.get_last_responce_time()
            })    

            preds.append(pred_label)
            if isinstance(y, int):
                trues.append(y)
            else:
                trues.append(y.item())           

    return response_times, trues, preds



def double(model_a, model_b, valset, threshold, score_function, device):
    response_times = []
    preds = []
    trues = []
    monitor = ResourceMonitor()
    second_model_usage= 0

    # Define score function
    if score_function == 'maxp':
        from utils.score_functions import max_probability
        score_fn = max_probability
    elif score_function == 'difference':
        from utils.score_functions import difference
        score_fn = difference
    elif score_function == 'entropy':
        from utils.score_functions import entropy
        score_fn = entropy
        
    input('RESET POWER METER AND PRESS ENTER...')

    if not score_function == 'entropy':
        with inference_mode():
            for _, (X, y) in tqdm(enumerate(valset), total=len(valset)):
            
                X = X.unsqueeze(dim=0).to(device)

                monitor.record_cpu_util()
                
                logits = model_a(X)

                score = score_fn(logits)

                if score < threshold:
                    second_model_usage += 1
                    logits = model_b(X)
                    # probs = softmax(logits, dim=1)

                pred_label = argmax(logits).item()

                monitor.record_cpu_util()

                response_times.append({
                    'cpu_util': monitor.get_last_cpu_util(),
                    'timestamp': monitor.get_last_timestamp(),
                    'response_time': monitor.get_last_responce_time()
                })

                preds.append(pred_label)
                if isinstance(y, int):
                    trues.append(y)
                else:
                    trues.append(y.item())

    else:
        with inference_mode():
            for _, (X, y) in tqdm(enumerate(valset), total=len(valset)):
                
                X = X.unsqueeze(dim=0).to(device)

                monitor.record_cpu_util()
                
                logits = model_a(X)

                score = score_fn(logits)

                if score > threshold:                    
                    logits = model_b(X)
                    second_model_usage += 1
                    # probs = softmax(logits, dim=1)

                pred_label = argmax(logits).item()

                monitor.record_cpu_util()

                response_times.append({
                    'cpu_util': monitor.get_last_cpu_util(),
                    'timestamp': monitor.get_last_timestamp(),
                    'response_time': monitor.get_last_responce_time()
                })
                
                preds.append(pred_label)
                if isinstance(y, int):
                    trues.append(y)
                else:
                    trues.append(y.item())
    
    return response_times, trues, preds, second_model_usage / len(valset)

def double_ps(model_a, model_b, valset, threshold, score_function, device):
    response_times = []
    preds = []
    trues = []
    monitor = ResourceMonitor()
    second_model_usage = 0
    

    # Define score function
    if score_function == 'maxp':
        from utils.score_functions import max_probability
        score_fn = max_probability
    elif score_function == 'difference':
        from utils.score_functions import difference
        score_fn = difference
    elif score_function == 'entropy':
        from utils.score_functions import entropy
        score_fn = entropy

    input('RESET POWER METER AND PRESS ENTER...')

    if not score_function == 'entropy':
        with inference_mode():
            for _, (X, y) in tqdm(enumerate(valset), total=len(valset)):

                X = X.unsqueeze(dim=0).to(device)

                monitor.record_cpu_util()

                logits_a = model_a(X)

                score_a = score_fn(logits_a)

                if score_a < threshold:
                    logits_b = model_b(X)
                    second_model_usage += 1
                    score_b = score_fn(logits_b)

                    if score_b > score_a:
                        pred_label = argmax(logits_b).item()
                    else:
                        pred_label = argmax(logits_a).item()
                else:
                    pred_label = argmax(logits_a).item()

                monitor.record_cpu_util()

                response_times.append({
                    'cpu_util': monitor.get_last_cpu_util(),
                    'timestamp': monitor.get_last_timestamp(),
                    'response_time': monitor.get_last_responce_time()
                })

                preds.append(pred_label)
                if isinstance(y, int):
                    trues.append(y)
                else:
                    trues.append(y.item())    
    else:
        with inference_mode():
            for _, (X, y) in tqdm(enumerate(valset), total=len(valset)):
                # X, y = X.unsqueeze(dim=0).to(device), tensor(y).unsqueeze(dim=0).to(device)
                X = X.unsqueeze(dim=0).to(device)

                monitor.record_cpu_util()

                logits_a = model_a(X)

                score_a = score_fn(logits_a)

                if score_a > threshold:
                    logits_b = model_b(X)
                    second_model_usage += 1
                    score_b = score_fn(logits_b)

                    if score_b < score_a:
                        pred_label = argmax(logits_b).item()
                    else:
                        pred_label = argmax(logits_a).item()
                else:
                    pred_label = argmax(logits_a).item()

                monitor.record_cpu_util()

                response_times.append({
                    'cpu_util': monitor.get_last_cpu_util(),
                    'timestamp': monitor.get_last_timestamp(),
                    'response_time': monitor.get_last_responce_time()
                })
            
                preds.append(pred_label)
                if isinstance(y, int):
                    trues.append(y)
                else:
                    trues.append(y.item())   
    
    return response_times, trues, preds, second_model_usage / len(valset)


def double_ps_mem(model_a, model_b, valset, threshold, score_function, device, memory):
    print("using memory")
    response_times = []
    preds = []
    trues = []
    monitor = ResourceMonitor()
    img_hash_lib = {}
    second_model_usage = 0

    # Define score function
    if score_function == 'maxp':
        from utils.score_functions import max_probability
        score_fn = max_probability
    elif score_function == 'difference':
        from utils.score_functions import difference
        score_fn = difference
    elif score_function == 'entropy':
        from utils.score_functions import entropy
        score_fn = entropy

    if memory == 'invariants':
        # from utils.im_fingerprint import inv_hash
        from utils.perceptual_hashing import complex_invariants_hash_addition_float
        hash_fn = complex_invariants_hash_addition_float
    elif memory == 'dhash':
        from utils.perceptual_hashing import dhash
        hash_fn = dhash

    input('RESET POWER METER AND PRESS ENTER...')

    if not score_function == 'entropy':
        with inference_mode():
            for _, (X, y, z) in tqdm(enumerate(valset), total=len(valset)):

                X = X.unsqueeze(dim=0).to(device)

                monitor.record_cpu_util()

                # calculate image hash
                image_hash = hash_fn(z)

                # search for hash matches in hash set image library
                hash_label = img_hash_lib.get(image_hash, None)
                if hash_label is not None:
                    pred_label = hash_label
                else:
                    logits_a = model_a(X)
                    score_a = score_fn(logits_a)

                    if score_a < threshold:
                        logits_b = model_b(X)
                        second_model_usage += 1
                        score_b = score_fn(logits_b)

                        if score_b > score_a:
                            pred_label = argmax(logits_b).item()
                        else:
                            pred_label = argmax(logits_a).item()
                    else:
                        pred_label = argmax(logits_a).item()

                # Assign predicted label from ANNs to new image hash and append to hash db
                img_hash_lib[image_hash] = pred_label

                monitor.record_cpu_util()

                response_times.append({
                    'cpu_util': monitor.get_last_cpu_util(),
                    'timestamp': monitor.get_last_timestamp(),
                    'response_time': monitor.get_last_responce_time()
                })

                preds.append(pred_label)
                if isinstance(y, int):
                    trues.append(y)
                else:
                    trues.append(y.item())   

    else:
        with inference_mode():
            for _, (X, y, z) in tqdm(enumerate(valset), total=len(valset)):

                X = X.unsqueeze(dim=0).to(device)

                monitor.record_cpu_util()

                # calculate image hash
                image_hash = hash_fn(z)

                # search for hash matches in hash set image library
                hash_label = img_hash_lib.get(image_hash, None)
                if hash_label is not None:
                    pred_label = hash_label
                else:
                    logits_a = model_a(X)
                    score_a = score_fn(logits_a)

                    if score_a > threshold:
                        logits_b = model_b(X)
                        second_model_usage += 1
                        score_b = score_fn(logits_b)

                        if score_b < score_a:
                            pred_label = argmax(logits_b).item()
                        else:
                            pred_label = argmax(logits_a).item()
                    else:
                        pred_label = argmax(logits_a).item()                

                # Assign predicted label from ANNs to new image hash and append to hash db
                img_hash_lib[image_hash] = pred_label

                monitor.record_cpu_util()

                response_times.append({
                    'cpu_util': monitor.get_last_cpu_util(),
                    'timestamp': monitor.get_last_timestamp(),
                    'response_time': monitor.get_last_responce_time()
                })

                preds.append(pred_label)
                if isinstance(y, int):
                    trues.append(y)
                else:
                    trues.append(y.item())   


    return response_times, trues, preds, second_model_usage / len(valset)



def double_oracle(model_a, model_b, valset, device):
    print("Double oracle")
    response_times = []
    preds = []
    trues = []
    monitor = ResourceMonitor()
    second_model_usage = 0

    from torch import softmax

    input('RESET POWER METER AND PRESS ENTER...')

    with inference_mode():
        for _, (X, y) in tqdm(enumerate(valset), total=len(valset)):

            X = X.unsqueeze(dim=0).to(device)

            monitor.record_cpu_util()
            
            logits_a = model_a(X)
            probs_a = softmax(logits_a, dim=1)
            pred_label = argmax(probs_a).item()

            if pred_label != y.item():
                logits_b = model_b(X)
                second_model_usage += 1
                probs_b = softmax(logits_b, dim=1)
                pred_label = argmax(probs_b).item()

            monitor.record_cpu_util()

            response_times.append({
                'cpu_util': monitor.get_last_cpu_util(),
                'timestamp': monitor.get_last_timestamp(),
                'response_time': monitor.get_last_responce_time()
            })

            preds.append(pred_label)
            if isinstance(y, int):
                trues.append(y)
            else:
                trues.append(y.item())   

    return response_times, trues, preds, second_model_usage / len(valset)