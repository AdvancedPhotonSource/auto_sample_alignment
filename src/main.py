import click

from align import Alignment
from pin_tracking import tracking

@click.command()
@click.option('-i', '--image_file', required=True)
@click.option('-a', '--algorithm', default='pin')
@click.option('-b', '--binary_threshold', default=0.4)
@click.option('-cl', '--canny_low', default=0.5)
@click.option('-ch', '--canny_high', default=0.8)
@click.option('-wh', '--window_height', default=200)
@click.option('-ww', '--window_width', default=200)
@click.option('--tracking/--no-tracking', default=False)
@click.option('-g', '--gap', default=100)
@click.option('-r', '--repeat', default=1)
@click.option('--transpose/--no-transpose', default=False)
@click.option('--debug/--no-debug', default=False)
@click.option('--quiet/--no-quiet', default=True)

def driver(image_file, 
          algorithm,
          binary_threshold, 
          canny_low, 
          canny_high, 
          window_height,
          window_width,
          tracking,
          gap, 
          repeat,
          transpose, 
          debug, 
          quiet):
    align = Alignment(image_file)
    params = {
        'binary_threshold': binary_threshold,
        'canny_thresh_low': canny_low,
        'canny_thresh_high': canny_high,
        'window_height': window_height,
        'window_width' : window_width,
        'gap': gap,
        'transpose': transpose,
        'debug': debug,
        'quiet': quiet
    }
    
    if tracking:
        tracking(params, algorithm, repeat)
    elif algorithm:
        print(align.compute_center(algorithm, params));

if __name__ == '__main__':
    driver()