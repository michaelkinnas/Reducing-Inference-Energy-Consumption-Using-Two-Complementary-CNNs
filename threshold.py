from torch.utils import data
from utils.scorefunctions import max_probability, difference, entropy
from utils.datasets import ImageNet, CIFAR10, INTEL, FashionMNIST
from utils.models_lists import imagenet_models, cifar10_models
from utils.heuristics import threshold_search
from torch import cuda, hub, inference_mode, argmax, tensor, load
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale
from torchvision.models import get_model
from tqdm import tqdm
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from multiprocessing.pool import Pool

'''
This script will run the hypermarameter search on the provided validation dataset
and then calculate the optimal threshold hyperparamter for each combination of 
score functions with and without the post-check mechanism, for max accuracy.
'''

def main():
    parser = ArgumentParser()

    parser.add_argument("-D", "--dataset", help="Define which dataset models to use.", choices=['cifar10', 'imagenet','intel', 'fashionmnist'], default='cifar-10', required=True)
    parser.add_argument("-f", "--dataset-root", help="The root file path of the validation or test dataset. (e.g. For CIFAR-10 the directory containing the 'cifar-10-batches-py' folder, etc.)", required=True)
    parser.add_argument("-m1", "--model1", help="The first model, required.", required=True)
    parser.add_argument("-m2", "--model2", help="The second model, required.", required=True)    
    parser.add_argument("-t", "--train", help="Only valid for the CIFAR-10 dataset. Define wether to use the training or test dataset.", default=False, action="store_true")
    parser.add_argument("-n", "--n_threshold_values", help="Define the number of threshold values to check between 0 and 1. Higher numbers will be slower. Default is 2000", type=int, default=2000)
    parser.add_argument("-w1", "--weights1", help="Optional. Directory of the '.pth' weights file for the first model.", default=None)
    parser.add_argument("-w2", "--weights2", help="Optional. Directory of the '.pth' weights file for the second model.", default=None)
    
    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'
    if device == 'cuda':
        cuda.empty_cache()

        # Setup parameters
    if args.dataset == 'cifar10':
        n_classes = 10

        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)) # cifar10
        ])

        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model1}', pretrained=True).to(device)
        model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model2}', pretrained=True).to(device)

        dataset = CIFAR10(root=args.dataset_root, train=args.train, transform=transform, seed=42)
        
        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)        
    elif args.dataset == 'imagenet':
        n_classes = 1000

        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) # imagenet
        ])

        model_a = get_model(args.model1, weights=imagenet_models[args.model1]).to(device)
        model_b = get_model(args.model2, weights=imagenet_models[args.model2]).to(device)

        dataset = ImageNet(root=args.dataset_root, transform=transform, seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    elif args.dataset == 'intel':
        n_classes = 10

        transform = Compose([
            Resize((32,32)),
            ToTensor(),    
            Normalize((0.4302, 0.4575, 0.4539), (0.2386, 0.2377, 0.2733))
        ])

        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model1}', pretrained=True).to(device)
        model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model2}', pretrained=True).to(device)

        split = "test" if args.train == True else "train"

        dataset = INTEL(root=args.dataset_root, split=split, transform=transform, seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)  


    elif args.dataset == 'fashionmnist':
        n_classes = 10

        transform = Compose([  
            Resize((32,32)),
            Grayscale(num_output_channels=3),
            ToTensor(),    
            Normalize((0.2856, 0.2856, 0.2856), (0.3385, 0.3385, 0.3385))
        ])

        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model1}', pretrained=True).to(device)
        model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model2}', pretrained=True).to(device)

        dataset = FashionMNIST(root=args.dataset_root, split=split, transform=transform, seed=42)

        BATCH_SIZE=64

        dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, train=args.train)
        
    entropy_norm_factor = entropy(tensor([1/n_classes for _ in range(n_classes)]).unsqueeze(dim=0))

    if args.dataset in ['cifar10', 'intel', 'fashionmnist']:
        if args.weights1 is not None:
            model_a = load(f'{args.weights1}', map_location="cpu", weights_only=False).to(device)
        if args.weights2 is not None:
            model_b = load(f'{args.weights2}', map_location="cpu", weights_only=False).to(device)

    model_a.eval()
    model_b.eval()

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
        for ret in pool.starmap(threshold_search, maxp_splits):
            maxp_results += ret[0]
            maxp_ps_results += ret[1]

    with Pool(n_threads) as pool:
        print("Searching parameter with difference, please wait......")
        for ret in pool.starmap(threshold_search, diff_splits):
            diff_results += ret[0]
            diff_ps_results += ret[1]
    
    with Pool(n_threads) as pool:
        print("Searching parameter with entropy, please wait......")
        for ret in pool.starmap(threshold_search, entropy_splits):
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