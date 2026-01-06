import argparse

from utils import dist_util, logger
from utils.featurization import construct_loader
from utils.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    data_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from utils.training_utils import TrainLoop

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )

    logger.log("creating data loader...")
    train_loader, val_loader = construct_loader(
        data_dir=args.data_dir,
        split_path=args.split_path,
        cache=args.cache,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_train_mols=args.limit_train_mols,
        modes=("train", "val"),
    )

    logger.log("training...")
    TrainLoop(
      model=model,
      diffusion=diffusion,
      data=train_loader,
      batch_size=args.batch_size,
      microbatch=args.microbatch,
      lr=args.lr,
      adam_beta1=args.adam_beta1,
      adam_beta2=args.adam_beta2,
      warmup_steps=args.warmup_steps,
      ema_rate=args.ema_rate,
      epochs=args.epochs,
      log_interval=args.log_interval,
      save_interval=args.save_interval,
      resume_checkpoint=args.resume_checkpoint,
      use_fp16=args.use_fp16,
      fp16_scale_growth=args.fp16_scale_growth,
      schedule_sampler=schedule_sampler,
      weight_decay=args.weight_decay,).run_loop()

def create_argparser():
    defaults = dict(
        # training
        schedule_sampler="uniform",
        lr=3e-4,  
        min_lr=3e-5,  
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        lr_schedule="cosine", 
        warmup_steps=1000,
        epochs=1000,
        batch_size=128,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=2,
        resume_checkpoint="",
        use_fp16=False,
        use_bf16=True, 
        fp16_scale_growth=1e-3,
    )

    defaults.update(data_defaults())
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
