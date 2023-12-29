from mmengine.runner import Runner
from mmengine import Config


def main():
    cfg = Config.fromfile('./configs/1ours_rcnn/base_rescale_collage+.py')
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    for batch in dataloader:
        print(batch)
        print(type(batch))
        break


if __name__ == '__main__':
    main()
