import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os
from cs336_basics.transformer import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.funciton import load_checkpoint, save_checkpoint, data_loading, gradient_clipping, learning_rate_schedule
from cs336_basics.loss import cross_entropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--validation_path", type=str, required=True)
    parser.add_argument("--fig_path", type=str, default="figs")
    parser.add_argument("--seed", type=int, default=None)

    # model hp
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")

    # adamw hp
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--betas",
        type=lambda s: tuple(map(float, s.split(","))),
        default=(0.9, 0.999),
        help="Adam betas, e.g. '0.9,0.999'",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay")

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="0 表示自动估算")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_times", type=int, default=64)

    # grad clip
    parser.add_argument("--clip_grad", type=float, default=1.0)

    # lr schedule
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--warmup_t", type=int, default=0)
    parser.add_argument("--cos_cycle_t", type=int, default=0)
    parser.add_argument("--lr_min", type=float, default=1e-5)

    # checkpointing
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    
    args = parser.parse_args()
    
    os.makedirs(args.fig_path, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    train_data = np.memmap(args.train_path, np.int32, mode="r")
    validation_data = np.memmap(args.validation_path, np.int32, mode="r")
    
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        device,
    )
    
    optimizer = AdamW(model.parameters(), args.lr, args.betas, args.eps, args.weight_decay)
    
    step = 0
    if args.resume:
        step = load_checkpoint(args.checkpoint_path, model, optimizer)
    
    #估算步数
    if args.steps_per_epoch <= 0:
        est = len(train_data) / (args.batch_size * args.context_length)
        args.steps_per_epoch = est
        
    train_losses = []
    train_lrs = []
    val_losses_history = []
    
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        
        for step_in_epoch in range(args.steps_per_epoch):
            step += 1
            print(f"step:{step}")
            inputs, targets = data_loading(
                train_data, args.batch_size, args.context_length, device
            )
            
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if args.clip_grad and args.clip_grad > 0.0:
                gradient_clipping(model.parameters(), args.clip_grad)
                
            if args.use_scheduler and args.cos_cycle_t > 0:
                lr_cur = learning_rate_schedule(
                    t = step,
                    lr_max=args.lr,
                    lr_min=args.lr_min,
                    warmup_t=args.warmup_t,
                    cos_cycle_t=args.cos_cycle_t,
                )
                for g in optimizer.param_groups:
                    g["lr"] = lr_cur
                    
            optimizer.step()
            
            running_loss += loss.item()
            
            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                ppl = float(torch.exp(torch.tensor(avg_loss)))
                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(
                    f"[epoch {epoch} step {step}] "
                    f"loss={avg_loss:.4f} ppl={ppl:.2f} lr={cur_lr:.6g} "
                    f"elapsed={elapsed:.1f}s"
                )
                train_losses.append(avg_loss)
                train_lrs.append(cur_lr)
                running_loss = 0.0
                t0 = time.time()
                
            if step % args.ckpt_interval == 0:
                verpath = f"{args.checkpoint_path}-{step}"
                save_checkpoint(model, optimizer, step, verpath)
                
                fig, axes = plt.subplots(2, 1, figsize=(8,10))
                iters = np.arange(1, len(train_losses) + 1) * args.log_interval
                axes[0].plot(iters, train_losses, label="train loss")
                axes[0].set_title("train loss")
                axes[0].set_xlabel("steps")
                axes[0].set_ylabel("loss")
                axes[0].legend()
                axes[1].plot(iters, train_lrs, label="train lr")
                axes[1].set_title("train lr")
                axes[1].set_xlabel("steps")
                axes[1].set_ylabel("lr")
                axes[1].legend()
                plt.tight_layout()
                plt.savefig(os.path.join(args.fig_path, "training_curves.png"))
                
                print(f"[ckpt] saved to {verpath} at step {step}")
                
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(args.val_times):
                xb, yb = data_loading(
                    validation_data,
                    args.batch_size,
                    args.context_length,
                    device
                )
                logits = model(xb)
                vloss = cross_entropy(logits, yb)
                val_losses.append(vloss.item())
        

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_ppl = float(np.exp(val_loss)) if np.isfinite(val_loss) else float("inf")
        val_losses_history.append(val_loss)
        print(f"[val epoch {epoch}] loss={val_loss:.4f} ppl={val_ppl:.2f}")
        
    epochs = np.arange(len(val_losses_history))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, val_losses_history, label="val loss")
    plt.title("val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.fig_path, "val_curves.png"))
    
    final_ckpt = f"{args.checkpoint_path}-final"
    save_checkpoint(model, optimizer, step, final_ckpt)
        
if __name__ == "__main__":
    main()      