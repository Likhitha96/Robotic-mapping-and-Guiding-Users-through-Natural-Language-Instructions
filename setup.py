from setuptools import setup

setup(
    name='Robotic-navigation-guide',
    version='1',
    packages=['tests', 'seq2seq', 'seq2seq.loss', 'seq2seq.util', 'seq2seq.optim', 'seq2seq.models', 'seq2seq.dataset',
              'seq2seq.trainer', 'seq2seq.evaluator'],
    url='',
    license='IIIT Bangalore',
    author='likhita',
    author_email='likhitha@aganitha.ai',
    description='Robotic navigation and generating instructions'
)
