import argparse


def default_argument_parser():
    """
    Create a parser with some common arguments.

    Returns:
        argparse.ArgumentParser:
    """

    parser = argparse.ArgumentParser("Joseph's detection")

    parser.add_argument("-n", "--experiment-name", type=str)
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--checkpoint", default=None, type=str, help="checkpoint file")
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")

    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument("--fp16", default=False, action="store_true", help="Adopting mix precision training.")
    parser.add_argument("--cache", dest="cache", default=False, action="store_true",
                        help="Caching images to RAM for faster training.")

    parser.add_argument("--limit_train_batches", default=2, type=int, help="")
    parser.add_argument("--limit_val_batches", default=1, type=int, help="")
    """
    Trainer parameters: 
        accelerator="gpu", 
        devices=2, 
        accumulate_grad_batches=1, accumulate_grad_batches={5: 3, 10: 20}
            no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
        amp_backend="native" or "apex"
            PyTorch built-in AMP or NVIDIA Apex
        amp_level='O2'
            Only Apex: Recognized opt_levels are "O0", "O1", "O2", and "O3".
        auto_scale_batch_size=None
            Automatically tries to find the largest batch size that fits into memory, before any training.
        auto_select_gpus=False
        auto_lr_find=False
            call tune to find the lr: trainer.tune(model)
        benchmark=False
            This flag is likely to increase the speed of your system if your input sizes don’t change.
        deterministic=False
            Slower but ensures reproducibility
        callbacks
        check_val_every_n_epoch=10
        default_root_dir=os.getcwd()
        enable_checkpointing=True
        limit_train_batches=0.25 (10)
            run through only 25% of the test set each epoch or int 10 batches.
        limit_val_batches
        log_every_n_steps=50
        max_epochs
            If both max_epochs and max_steps aren’t specified, max_epochs will default to 1000.
        max_steps
        min_epochs
        min_steps
        max_time
            max_time={"days": 1, "hours": 5}, max_time="00:12:00:00" (12 hours)
        num_nodes
            Number of GPU nodes for distributed training.
        overfit_batches
        precision
        progress_bar_refresh_rate=1
             Lightning will set it to 20 in Google COLAB.
        enable_progress_bar=True
        reload_dataloaders_every_n_epochs
            Set to a postive integer to reload dataloaders every n epochs.
        replace_sampler_ddp=True
            Make sure each GPU or TPU sees the appropriate part of your data. 
        strategy
            Supports passing different training strategies with aliases (ddp, ddp_spawn, etc)
        sync_batchnorm
            Enable synchronization between batchnorm layers across all GPUs.
        tpu_cores
            How many TPU cores to train on (1 or 8).
        weights_save_path
    """


    return parser