import random
import shutil
from pathlib import Path

import click


@click.command()
@click.option('--out-path', required=True, 
              type=click.Path(file_okay=False, exists=True))
@click.option('--path', required=True,
              type=click.Path(file_okay=False, exists=True))
@click.option('--train-prob', type=float, default=0.85)
@click.option('--force-rm/--no-force-rm', default=False)
def main(**args):
    origin_path = Path(args['path'])
    out_path = Path(args['out_path'])
    train_prob = args['train_prob']

    out_sets = dict(train=out_path / 'train', test=out_path / 'test')
    for s in out_sets.values():
        if args['force_rm'] and s.exists():
            shutil.rmtree(str(s))
        s.mkdir(exist_ok=True)
    
    for f in origin_path.glob('*.json'):

        out_set = out_sets['train' if random.random() < train_prob else 'test']
        fname = f.stem + '.json'
        out_path = out_set / fname
        shutil.copy(str(f), str(out_path))


if __name__ == "__main__":
    main()

