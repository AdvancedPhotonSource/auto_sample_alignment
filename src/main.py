import click

from align import Alignment

@click.command()
@click.option('-i', '--image_file')
@click.option('--debug/--no-debug', default=True)
def driver(image_file, debug):
    align = Alignment(image_file)
    params = {
        'debug' = debug
    }
    align.align(params)

if __name__ == '__main__':
    driver()