import pandas as pd
import numpy as np
from torch.utils import data
from Deliverable.utils.scorefunctions import max_probability, difference, entropy
from utils.datasets import ImageNetC, CIFAR10C
from utils.models_lists import imagenet_models, cifar10_models
from utils.heuristics import heuristic_search_process
from torch import cuda, hub, inference_mode, argmax, tensor
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm.auto import tqdm
from torchvision.models import get_model
from argparse import ArgumentParser
from multiprocessing.pool import Pool

'''
This script will run the hypermarameter search on the provided validation dataset
and then calculate the optimal threshold hyperparamter for each combination of 
score functions with and without the post-check mechanism, for max accuracy.
'''

def main():
    parser = ArgumentParser()

    parser.add_argument("-m1", "--model1", help="The first model, required. This parameter will set which dataset to use (CIFAR10 or ImageNet)", required=True)   
    parser.add_argument("-m2", "--model2", help="The second model, required. ", required=True)    
    parser.add_argument("-f", "--filepath", help="The path of the correspondig CIFAR-10 or ImageNet validation dataset.", required=True)
    parser.add_argument("-t", "--train", help="Only valid for the CIFAR-10 dataset. Define wether to use the training or test dataset.", default=False, action="store_true")
    parser.add_argument("-n", "--n_threshold_values", help="Define the number of threshold values to check between 0 and 1. Higher numbers will be slower. Default is 2000", type=int, default=2000)
    
    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'
    if device == 'cuda':
        cuda.empty_cache()    

    # Setup parameters
    if args.model1 in cifar10_models:
        n_classes = 10

        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)) # cifar10
        ])

        DATASET = 'CIFAR10'

        MODEL_A = args.model1
        MODEL_B = args.model2
    
        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_A}', pretrained=True).to(device)
        model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{MODEL_B}', pretrained=True).to(device)

        model_a.eval()
        model_b.eval()

        dataset = CIFAR10C(root=args.filepath, train=args.train, transform=transform, random_seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)        
    else:
        n_classes = 1000

        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) # imagenet
        ])

        DATASET = 'ImageNet'

        MODEL_A = args.model1
        MODEL_B = args.model2

        model_a = get_model(MODEL_A, weights=imagenet_models[MODEL_A]).to(device)
        model_b = get_model(MODEL_B, weights=imagenet_models[MODEL_B]).to(device)

        model_a.eval()
        model_b.eval()

        dataset = ImageNetC(root=args.filepath, transform=transform, random_seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
    entropy_norm_factor = entropy(tensor([1/n_classes for _ in range(n_classes)]).unsqueeze(dim=0))

    # Run inference on validation dataset for both models
    report = []
    print("\nGetting predictions from selected models please wait...")
    with inference_mode():
        for X, y in tqdm(dataloader):
            X = X.to(device)

            preds_a = model_a(X)
            preds_b = model_b(X)

            for ya, yb, yt in zip(preds_a, preds_b, y):

                report.append({
                    'classification_a': argmax(ya).item(),
                    'max_prob_a' : max_probability(ya.unsqueeze(dim=0)),
                    'difference_a' : difference(ya.unsqueeze(dim=0)),
                    'entropy_a' : entropy(ya.unsqueeze(dim=0)) / entropy_norm_factor,
                    'classification_b': argmax(yb).item(),
                    'max_prob_b' : max_probability(yb.unsqueeze(dim=0)),
                    'difference_b' : difference(yb.unsqueeze(dim=0)),
                    'entropy_b' : entropy(yb.unsqueeze(dim=0)) / entropy_norm_factor,
                    'true': yt.item()
                })
    
    df = pd.DataFrame(report)

    threshold_params = np.linspace(0, 1, args.n_threshold_values)

    n_threads = 16

    step = args.n_threshold_values // n_threads

    maxp_splits = []
    diff_splits = []
    entropy_splits = []
    for i in range(n_threads):
        maxp_splits.append((df['true'], df['classification_a'], df['classification_b'], df['max_prob_a'], df['max_prob_b'], threshold_params[i * step : (i+1) * step], False))
        diff_splits.append((df['true'], df['classification_a'], df['classification_b'], df['difference_a'], df['difference_b'], threshold_params[i * step : (i+1) * step], False))
        entropy_splits.append((df['true'], df['classification_a'], df['classification_b'], df['entropy_a'], df['entropy_b'], threshold_params[i * step : (i+1) * step], True))

    maxp_results = []
    maxp_ps_results = []
    diff_results = []
    diff_ps_results = []
    entropy_results = []
    entropy_ps_results = []

    # Multi processing    
    with Pool(n_threads) as pool:
        print("\nSearching parameter with max probability, please wait...")
        for ret in pool.starmap(heuristic_search_process, maxp_splits):
            maxp_results += ret[0]
            maxp_ps_results += ret[1]

    with Pool(n_threads) as pool:
        print("Searching parameter with difference, please wait......")
        for ret in pool.starmap(heuristic_search_process, diff_splits):
            diff_results += ret[0]
            diff_ps_results += ret[1]
    
    with Pool(n_threads) as pool:
        print("Searching parameter with entropy, please wait......")
        for ret in pool.starmap(heuristic_search_process, entropy_splits):
            entropy_results += ret[0]
            entropy_ps_results += ret[1]
   

    df_max_p = pd.DataFrame(maxp_results)
    df_diff = pd.DataFrame(diff_results)
    df_entropy = pd.DataFrame(entropy_results).iloc[::-1].reset_index(drop=True) # Reverse order for entropy
    df_max_p_ps = pd.DataFrame(maxp_ps_results)
    df_diff_ps = pd.DataFrame(diff_ps_results)
    df_entropy_ps = pd.DataFrame(entropy_ps_results).iloc[::-1].reset_index(drop=True) # Reverse order for entropy


    print(f"\nFound threshold parameters for {args.model1}, {args.model2}\n")
    print(f"--------------------------------Without post-check-----------------------------")
    print(f"Max Probability, Threshold: {df_max_p.iloc[df_max_p['accuracy'].idxmax()].iat[0]:.4f}, with achieved accuracy of {df_max_p.iloc[df_max_p['accuracy'].idxmax()].iat[1] / len(dataset) * 100:.2f}%")
    print(f"Difference, Threshold: {df_diff.iloc[df_diff['accuracy'].idxmax()].iat[0]:.4f}, with achieved accuracy of {df_diff.iloc[df_diff['accuracy'].idxmax()].iat[1] / len(dataset)* 100:.2f}%")
    print(f"Entropy, Threshold: {df_entropy.iloc[df_entropy['accuracy'].idxmax()].iat[0]:.4f}, with achieved accuracy of {df_entropy.iloc[df_entropy['accuracy'].idxmax()].iat[1] / len(dataset)* 100:.2f}%")
    
    print(f"\n--------------------------------With post-check--------------------------------")
    print(f"Max Probability, Threshold: {df_max_p_ps.iloc[df_max_p_ps['accuracy'].idxmax()].iat[0]:.4f}, with achieved accuracy of {df_max_p_ps.iloc[df_max_p_ps['accuracy'].idxmax()].iat[1] / len(dataset)* 100:.2f}%")
    print(f"Difference, Threshold: {df_diff_ps.iloc[df_diff_ps['accuracy'].idxmax()].iat[0]:.4f}, with achieved accuracy of {df_diff_ps.iloc[df_diff_ps['accuracy'].idxmax()].iat[1] / len(dataset)* 100:.2f}%")
    print(f"Entropy, Threshold: {df_entropy_ps.iloc[df_entropy_ps['accuracy'].idxmax()].iat[0]:.4f}, with achieved accuracy of {df_entropy_ps.iloc[df_entropy_ps['accuracy'].idxmax()].iat[1] / len(dataset)* 100:.2f}%")


if __name__ == '__main__':
    main()