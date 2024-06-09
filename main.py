'''
Main inference workload script
v1.3.1
'''
# from sys import path
# path.append('../')

from argparse import ArgumentParser 
from torch import cuda
from os import system
from playsound import playsound
from utils.workload_fns import write_results
from utils.models_lists import cifar10_models


def main():
    parser = ArgumentParser()

    parser.add_argument("-m1", "--model1", help="The first model, required. This parameter will set which dataset to use (CIFAR10 or ImageNet)", required=True)   
    parser.add_argument("-m2", "--model2", help="The second model.", default=None, required=False)
    parser.add_argument("-i", "--filepath", help="The path of the corresponding CIFAR-10 or ImageNet validation or test dataset.", required=True)
    parser.add_argument("-s", "--scorefn", help="Score function to use.", choices=['maxp', 'difference', 'entropy', 'oracle'], default=None, required=False)
    parser.add_argument("-t", "--threshold", help="The threshold value to use for the threshold check.", type=float, required=False)
    parser.add_argument("-p", "--postcheck", help="Enable post check. Default is false.", default=False, action="store_true")
    parser.add_argument("-m", "--memory", help="Enable memory component. Default is None.", choices=['dhash', 'invariants'], default=None)
    parser.add_argument("-d", "--duplicates", help="Set the percentage of the original training set for duplication. Default is 0 (No duplicates). Range (0-1]", type=float, default=0)
    parser.add_argument("-r", "--rotations", help="Set the percentage of the duplicated samples to apply random rotations or flips if a --duplicates value is given. Default is 0. Range (0-1]", type=float, default=0)
    parser.add_argument("-f", "--finish", help="What to do when finished. Default is shutdown.", choices=['shutdown', 'alarm'], default="shutdown")

    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'

    if args.model2 and not args.scorefn or args.model2 and not args.threshold:
        raise ValueError("Score function and/or threshold value must be provided if model2 is used.")


    if args.model1 in cifar10_models:
        from torch import hub
        from utils.datasets import CIFAR10C
        from torchvision.transforms import Compose, ToTensor, Normalize

        transform = Compose([
                ToTensor(),
                Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201)) # CIFAR10
            ])
        
        print("Loading dataset...")

        if args.memory:
            valset = CIFAR10C(root=args.filepath, train=False, return_numpy=True, transform=transform, duplicate_ratio=args.duplicates, transform_prob=args.rotations, random_seed=42)
        else:
            valset = CIFAR10C(root=args.filepath, train=False, return_numpy=False, transform=transform, duplicate_ratio=args.duplicates, transform_prob=args.rotations, random_seed=42)

        print("Loading models...")
        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model1}', pretrained=True).to(device)
        model_a.eval()
        if args.model2 != None:
            if not args.model2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model2}', pretrained=True).to(device)
            model_b.eval()
    else:
        from torchvision.models import get_model
        from utils.datasets import ImageNetVal
        from torchvision.transforms import Compose, ToTensor, Normalize
        from utils.models_lists import imagenet_models

        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), #ImageNet
        ])
        print("Loading dataset...")
        if args.memory:
            valset = ImageNetVal(root=args.filepath, transform=transform, return_numpy=True, duplicate_ratio=args.duplicates, transform_prob=args.rotations, random_seed=42)
        else:
            valset = ImageNetVal(root=args.filepath, transform=transform, return_numpy=False, duplicate_ratio=args.duplicates, transform_prob=args.rotations, random_seed=42)

        print("Loading models...")
        model_a = get_model(args.model1, weights=imagenet_models[args.model1]).to(device)
        model_a.eval()
        if args.model2 != None:
            if not args.model2 in imagenet_models.keys():
                raise ValueError("Second model must be from the same ImageNet dataset.")

            model_b = get_model(args.model2, weights=imagenet_models[args.model2]).to(device)
            model_b.eval()


    print("\n-------------Set parameters------------")
    # print(f"Dataset: {args.dataset}")

    if args.model2:
        print(f"Models: {args.model1}, {args.model2} on {device}")
    else:
        print(f'Model: {args.model1} on {device}')

    if args.threshold and args.scorefn:
        print(f"Using {args.scorefn} score function with {args.threshold} threshold value.")

    if args.postcheck:
        print("Post-check enabled")

    if args.memory:
        print(f"Using memory component: {args.memory}.")

    if args.duplicates:
        print(f"Using {args.duplicates * 100:.2f}% duplicated samples for a total of {len(valset)} samples.")

    

    if not args.model2:
        from utils.workload_fns import single
        response_times, correct = single(model_a, valset=valset, device=device)
    else:
        if args.scorefn == 'oracle':
            from utils.workload_fns import double_oracle
            response_times, correct = double_oracle(model_a, model_b, valset=valset, device=device)
        else:
            if not args.postcheck:
                from utils.workload_fns import double
                response_times, correct = double(model_a=model_a, model_b=model_b, valset=valset, threshold=args.threshold, score_function=args.scorefn, device=device)
            else:
                if not args.memory:
                    from utils.workload_fns import double_ps
                    response_times, correct = double_ps(model_a=model_a, model_b=model_b, valset=valset, threshold=args.threshold, score_function=args.scorefn, device=device)
                else:
                    from utils.workload_fns import double_ps_mem
                    response_times, correct = double_ps_mem(model_a=model_a, model_b=model_b, valset=valset, threshold=args.threshold, score_function=args.scorefn, memory=args.memory, device=device)
        
    write_results(args, response_times=response_times, correct=correct)

    if args.finish == 'shutdown':

        system('sudo shutdown now')
    elif args.finish == 'alarm':
        playsound('./alarm.wav')


if __name__ == '__main__':
    main()
