from sklearn.metrics import classification_report
from argparse import ArgumentParser 
from torch import cuda, load
from torch.nn import Module
from os import system
# from playsound import playsound
from utils.models_lists import cifar10_models
from pandas import DataFrame




def main():
    parser = ArgumentParser()

    parser.add_argument("-m1", "--model-1", help="The first model, required. This parameter will set which dataset to use (CIFAR10 or ImageNet)", required=True)   
    parser.add_argument("-m2", "--model-2", help="The second model.", default=None, required=False)
    parser.add_argument("-w1", "--weights-1", help="Optional. A file path to the first model's weights file.", default=None, required=False)
    parser.add_argument("-w2", "--weights-2", help="Optional. A file path to the second model's weights file.", default=None, required=False)
    parser.add_argument("-f", "--filepath", help="The path of the corresponding CIFAR-10 or ImageNet validation or test dataset.", required=True)
    parser.add_argument("-s", "--scorefn", help="Score function to use.", choices=['maxp', 'difference', 'entropy', 'oracle'], default=None, required=False)
    parser.add_argument("-t", "--threshold", help="The threshold value to use for the threshold check.", type=float, required=False)
    parser.add_argument("-p", "--postcheck", help="Enable post check. Default is false.", default=False, action="store_true")
    parser.add_argument("-m", "--memory", help="Enable memory component. Default is None.", choices=['dhash', 'invariants'], default=None)
    parser.add_argument("-d", "--duplicates", help="Set the percentage of the original training set for duplication. Default is 0 (No duplicates). Range [0-1]", type=float, default=0)
    parser.add_argument("-r", "--rotations", help="Set the percentage of the duplicated samples to apply random rotations or flips if a --duplicates value is given. Default is 0. Range [0-1]", type=float, default=0)
    # parser.add_argument("-e", "--end", help="What to do when finished. Default is shutdown.", choices=['shutdown', 'alarm'], default="alarm")
    parser.add_argument("-rp", "--root-password", help="If provided the password will be used to command the computer to shutdown after finishing. Else an alarm will sound.", required=None, default=None)

    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'

    if args.model_2 and not args.scorefn or args.model_2 and not args.threshold:
        raise ValueError("Score function and/or threshold value must be provided if model2 is used.")


    if args.model_1 in cifar10_models:
        from torch import hub
        from utils.datasets import CIFAR10C
        from torchvision.transforms import Compose, ToTensor, Normalize

        dataset = "CIFAR-10"

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
        model_a = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model_1}', pretrained=True).to(device)
        model_a.eval()
        if args.model_2 != None:
            if not args.model_2 in cifar10_models:
                raise ValueError("Second model must be from the same CIFAR10 dataset.")

            model_b = hub.load("chenyaofo/pytorch-cifar-models", model=f'cifar10_{args.model_2}', pretrained=True).to(device)
            model_b.eval()
    else:
        from torchvision.models import get_model
        from utils.datasets import ImageNetC
        from torchvision.transforms import Compose, ToTensor, Normalize
        from utils.models_lists import imagenet_models

        dataset = "ImageNet"

        transform = Compose([
            ToTensor(),
            Normalize(mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), #ImageNet
        ])
        print("Loading dataset...")
        if args.memory:
            valset = ImageNetC(root=args.filepath, transform=transform, return_numpy=True, duplicate_ratio=args.duplicates, transform_prob=args.rotations, random_seed=42)
        else:
            valset = ImageNetC(root=args.filepath, transform=transform, return_numpy=False, duplicate_ratio=args.duplicates, transform_prob=args.rotations, random_seed=42)

        print("Loading models...")
        model_a = get_model(args.model_1, weights=imagenet_models[args.model_1]).to(device)
        model_a.eval()
        if args.model_2 != None:
            if not args.model_2 in imagenet_models.keys():
                raise ValueError("Second model must be from the same ImageNet dataset.")

            model_b = get_model(args.model_2, weights=imagenet_models[args.model_2]).to(device)
            model_b.eval()


    # Load weights if provided
    if args.weights_1 is not None:
        loaded = load(args.weights_1, map_location="cpu", weights_only=False)
        # In case the whole nn.Module is saved
        if isinstance(loaded, Module):
            model_a = loaded
        # In case only the state dictionary of the module is saved
        else:
            model_a.load_state_dict(loaded)
        
    if args.weights_2 is not None:
        loaded = load(args.weights_2, map_location="cpu", weights_only=False)
        # In case the whole nn.Module is saved
        if isinstance(loaded, Module):
            model_b = loaded
        # In case only the state dictionary of the module is saved
        else:
            model_b.load_state_dict(loaded)


    print("\n-------------Set parameters------------")

    print(f"Dataset: {dataset}")

    if args.model_2:
        print(f"First Model: {args.model_1}, {sum(p.numel() for p in model_a.parameters()) / 1000000:.2f}M parameters.")
        print(f"Second Model: {args.model_2}, {sum(p.numel() for p in model_b.parameters())/ 1000000:.2f}M parameters.")
    else:
        print(f'Model: {args.model_1}, {sum(p.numel() for p in model_a.parameters())/ 1000000:.2f}M parameters.')

    print(f"Device: {device}")

    if args.threshold and args.scorefn:
        print(f"Score function: {args.scorefn}")
        print(f"Threshold parameter: {args.threshold}")

    if args.postcheck:
        print("Post-check enabled")

    if args.memory:
        print(f"Using memory component: {args.memory}.")

    if args.duplicates:
        print(f"Using {args.duplicates * 100:.2f}% duplicated samples for a total of {len(valset)} samples.")
    
    print("---------------------------------------")

    

    if not args.model_2:
        from utils.workload_fns import single
        response_times, trues, preds = single(model_a, valset=valset, device=device)
    else:
        if args.scorefn == 'oracle':
            from utils.workload_fns import double_oracle
            response_times, trues, preds = double_oracle(model_a, model_b, valset=valset, device=device)
        else:
            if not args.postcheck:
                from utils.workload_fns import double
                response_times, trues, preds = double(model_a=model_a, model_b=model_b, valset=valset, threshold=args.threshold, score_function=args.scorefn, device=device)
            else:
                if not args.memory:
                    from utils.workload_fns import double_ps
                    response_times, trues, preds = double_ps(model_a=model_a, model_b=model_b, valset=valset, threshold=args.threshold, score_function=args.scorefn, device=device)
                else:
                    from utils.workload_fns import double_ps_mem
                    response_times, trues, preds = double_ps_mem(model_a=model_a, model_b=model_b, valset=valset, threshold=args.threshold, score_function=args.scorefn, memory=args.memory, device=device)
        
    
    print("\n---Classification report---")
    print(classification_report(trues, preds, output_dict=False, zero_division=0, digits=4))
    df = DataFrame(response_times)
    df.drop(index=df.index[0:25], axis=0, inplace=True)
    print("\n---Response times---")
    print(f"Mean: {df['response_time'].mean() * 1000:.3f} ms")
    print(f"95th: {df['response_time'].quantile(0.95) * 1000:.3f} ms")
    print(f"99th: {df['response_time'].quantile(0.99) * 1000:.3f} ms")

    if args.root_password:
        system(f"echo {args.root_password} | sudo -S poweroff")
    else:
        # playsound('./alarm.wav')
        pass


if __name__ == '__main__':
    main()
