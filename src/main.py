import click

from align import Alignment

@click.command()
@click.option('-i', '--image_file', required=True)
@click.option('-b', '--binary_threshold', default=0.4)
@click.option('-c1', '--canny_low', default=0.5)
@click.option('-c2', '--canny_high', default=0.5)
@click.option('--transpose/--no-transpose', default=False)
@click.option('--debug/--no-debug', default=True)
def driver(image_file, binary_threshold, canny_low, canny_high, transpose, debug):
    align = Alignment(image_file)
    params = {
        'binary_threshold': binary_threshold,
        'canny_thresh1': canny_low,
        'canny_thresh2': canny_high,
        'transpose': transpose,
        'debug': debug
    }
    align.align(params)

if __name__ == '__main__':
    driver()