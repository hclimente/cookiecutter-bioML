from setuptools import setup, find_packages

with open("README.md", "r") as file:
	readme = file.read()

setup(
	name = '{{cookiecutter.project_name}}'
	packages = find_packages(),
	package_dir = {'{{cookiecutter.project_name}}': '{{cookiecutter.project_name}}'},
	version = '0.1',
	description = '{{cookiecutter.project_short_description}}',
	long_description=readme,
	long_description_content_type='text/markdown',
	author = '{{cookiecutter.full_name}}',
	author_email = '{{cookiecutter.email}}',
	license = 'GPL-3',
	url = 'https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_name}}',
	keywords = [],
	classifiers = [
        'Development Status :: 3 - Alpha',
		'Topic :: Scientific/Engineering :: Mathematics',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3 :: Only'],
	install_requires = [],
)
