"""Entry point to run MAF and CycleGAN experiments.
Usage examples:
  python run.py maf --mode train --epochs 2 --quick
  python run.py maf --mode generate --num_samples 4
  python run.py cyclegan --mode train --epochs 2 --quick
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader

from maf import MAF, train_maf, generate_images_maf
from datasets import CapsuleDataset, capsule_transform, ImageDataset, cyclegan_transform
from utils import visualize_samples, evaluate_anomaly_detection, plot_roc_curve, plot_score_distributions
from cyclegan import Generator, Discriminator, train_cyclegan, test_cyclegan


def run_maf(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = args.dataset if args.dataset else 'capsule/train/good'
    if not os.path.isdir(dataset_path):
        print(f"Capsule dataset not found at {dataset_path}. Use --quick to run a smoke test.")
        if args.quick:
            # create random dataset
            import torch
            class RandomDataset(torch.utils.data.Dataset):
                def __init__(self, n=10, dim=128*128*3):
                    self.n = n; self.dim = dim
                def __len__(self): return self.n
                def __getitem__(self, idx):
                    x = torch.randn(self.dim)
                    return x
            train_loader = DataLoader(RandomDataset(n=10), batch_size=3)
        else:
            return
    else:
        train_dataset = CapsuleDataset(root_dir=dataset_path, transform=capsule_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    input_dim = 128 * 128 * 3
    maf = MAF(input_dim=input_dim, num_blocks=args.num_blocks, hidden_dims=[512, 512])

    if args.mode == 'train':
        losses = train_maf(maf, train_loader, num_epochs=args.epochs, lr=args.lr, device=device)
        torch.save(maf.state_dict(), args.out if args.out else 'maf_final.pth')
        print('Training finished and model saved.')
    elif args.mode == 'generate':
        maf.load_state_dict(torch.load(args.model)) if args.model else None
        samples, gen_time = generate_images_maf(maf, num_images=args.num_samples, img_size=128, device=device)
        print(f"Generation time: {gen_time:.2f}s ({gen_time/args.num_samples:.2f}s per image)")
        visualize_samples([samples[i] for i in range(samples.shape[0])])
    elif args.mode == 'eval':
        if not args.model:
            print('Provide --model for evaluation')
            return
        maf.load_state_dict(torch.load(args.model, map_location=device))
        maf = maf.to(device)
        # prepare test loaders
        normal_path = 'capsule/test/good'
        anomaly_path = 'capsule/test/crack'
        if not os.path.isdir(normal_path) or not os.path.isdir(anomaly_path):
            print('Test datasets missing; run with --quick to perform a small synthetic evaluation')
            if args.quick:
                # synthetic evaluation
                normal_scores = torch.abs(torch.randn(50)).numpy()
                anomaly_scores = torch.abs(torch.randn(50) + 0.5).numpy()
                auroc, fpr, tpr = evaluate_anomaly_detection(normal_scores, anomaly_scores)
                print(f'AUROC (synthetic): {auroc:.4f}'); plot_roc_curve(fpr, tpr, auroc); plot_score_distributions(normal_scores, anomaly_scores)
            return
        from datasets import CapsuleDataset
        from torch.utils.data import DataLoader
        normal_loader = DataLoader(CapsuleDataset(normal_path, transform=capsule_transform), batch_size=8)
        anomaly_loader = DataLoader(CapsuleDataset(anomaly_path, transform=capsule_transform), batch_size=8)
        # compute scores
        import numpy as np
        def calc_scores(model, loader):
            model.eval(); scores = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    _, log_prob = model.forward(batch.view(batch.size(0), -1))
                    nll = -log_prob
                    scores.extend(nll.cpu().numpy())
            return np.array(scores)
        normal_scores = calc_scores(maf, normal_loader)
        anomaly_scores = calc_scores(maf, anomaly_loader)
        auroc, fpr, tpr = evaluate_anomaly_detection(normal_scores, anomaly_scores)
        print(f'AUROC: {auroc:.4f}')
        plot_roc_curve(fpr, tpr, auroc)
        plot_score_distributions(normal_scores, anomaly_scores)
    else:
        print('Unknown mode for maf')


def run_cyclegan(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_root = args.dataset if args.dataset else 'horse2zebra'
    if args.mode == 'train':
        trainA = os.path.join(dataset_root, 'trainA')
        trainB = os.path.join(dataset_root, 'trainB')
        if not os.path.isdir(trainA) or not os.path.isdir(trainB):
            print('Dataset not found. Use --quick to run a smoke training.')
            if args.quick:
                # quick smoke: random tensors
                class RDS(torch.utils.data.Dataset):
                    def __len__(self): return 20
                    def __getitem__(self, idx): return torch.randn(3, 128, 128)
                loaderA = DataLoader(RDS(), batch_size=2)
                loaderB = DataLoader(RDS(), batch_size=2)
            else:
                return
        else:
            loaderA = DataLoader(ImageDataset(os.path.join(dataset_root, 'trainA'), transform=cyclegan_transform), batch_size=args.batch_size, shuffle=True)
            loaderB = DataLoader(ImageDataset(os.path.join(dataset_root, 'trainB'), transform=cyclegan_transform), batch_size=args.batch_size, shuffle=True)

        G_AB = Generator(); G_BA = Generator(); D_A = Discriminator(); D_B = Discriminator()
        history = train_cyclegan(G_AB, G_BA, D_A, D_B, loaderA, loaderB, num_epochs=args.epochs, device=device, checkpoint_dir=args.checkpoint_dir)
        print('CycleGAN training finished')
    elif args.mode == 'test':
        testA = os.path.join(dataset_root, 'testA'); testB = os.path.join(dataset_root, 'testB')
        if not os.path.isdir(testA) or not os.path.isdir(testB):
            print('Test sets missing')
            return
        loaderA = DataLoader(ImageDataset(testA, transform=cyclegan_transform), batch_size=args.num_samples)
        loaderB = DataLoader(ImageDataset(testB, transform=cyclegan_transform), batch_size=args.num_samples)
        G_AB = Generator(); G_BA = Generator()
        if args.model:
            G_AB.load_state_dict(torch.load(args.model))
        (real_A, fake_B, rec_A), (real_B, fake_A, rec_B) = test_cyclegan(G_AB, G_BA, loaderA, loaderB, device=device, num_samples=args.num_samples)
        visualize_samples([real_A[0], fake_B[0], rec_A[0]], titles=['Real A', 'Fake B', 'Rec A'])
        visualize_samples([real_B[0], fake_A[0], rec_B[0]], titles=['Real B', 'Fake A', 'Rec B'])
    else:
        print('Unknown mode for cyclegan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='task')

    # MAF parser
    p_maf = subparsers.add_parser('maf')
    p_maf.add_argument('--mode', choices=['train', 'generate', 'eval'], required=True)
    p_maf.add_argument('--dataset', type=str, default=None)
    p_maf.add_argument('--batch_size', type=int, default=3)
    p_maf.add_argument('--epochs', type=int, default=100)
    p_maf.add_argument('--lr', type=float, default=1e-4)
    p_maf.add_argument('--num_blocks', type=int, default=7)
    p_maf.add_argument('--num_samples', type=int, default=5)
    p_maf.add_argument('--model', type=str, default=None)
    p_maf.add_argument('--out', type=str, default=None)
    p_maf.add_argument('--quick', action='store_true')
    p_maf.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p_maf.add_argument('--save_every', type=int, default=10)
    p_maf.add_argument('--resume', type=str, default=None)
    p_maf.add_argument('--use_scheduler', action='store_true')

    # CycleGAN parser
    p_cyc = subparsers.add_parser('cyclegan')
    p_cyc.add_argument('--mode', choices=['train', 'test'], required=True)
    p_cyc.add_argument('--dataset', type=str, default=None)
    p_cyc.add_argument('--batch_size', type=int, default=16)
    p_cyc.add_argument('--epochs', type=int, default=20)
    p_cyc.add_argument('--num_samples', type=int, default=5)
    p_cyc.add_argument('--model', type=str, default=None)
    p_cyc.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p_cyc.add_argument('--quick', action='store_true')

    args = parser.parse_args()
    if args.task == 'maf':
        run_maf(args)
    elif args.task == 'cyclegan':
        run_cyclegan(args)
    else:
        parser.print_help()
