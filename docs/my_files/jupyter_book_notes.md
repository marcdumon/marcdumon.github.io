# Jupyter Book Notes


## Setup
- git:
    ````bash
    git remote add origin https://github.com/marcdumon/test_jupyter_book
    ````
- Notebook tags:
  See: [myst-cheatsheet-code-cell-tags](https://jupyterbook.org/en/stable/reference/cheatsheet.html#myst-cheatsheet-code-cell-tags)
    - output_scroll
    - hide-input : Hide cell but the display the outputs
    - hide-output
    - hide-cell : Hides inputs and outputs of code cell

## Update the book  

- Add pages to `_toc.yaml`.

- To remove the `_build` directory:
    ````bash
    jupyter-book clean ./ --all
    ````
- To recreate the `_build` directory:
    ````bash
    jupyter-book build .
    ````
- To push the updated book to Github:   
    
    ````bash
    ghp-import -n -p -f _build/html
    ````

    BTW: Authentication via password doesn't work anymore. Instead, generate a [Personal access tokens on github](https://github.com/settings/tokens), and then   
    update /media/Development/0_jupyter_book/.git/config:  
    [remote "origin"]   
	url = https://marcdumon:<TOKEN>@github.com/marcdumon/marcdumon.github.io.git    
	fetch = +refs/heads/*:refs/remotes/origin/*    


## Default configuration files
- `_config.yml`
    ````yaml
    # Add a bibtex file so that we can create citations
    bibtex_bibfiles:
    - references.bib

    # Launch button settings
    launch_buttons:
    notebook_interface        : classic  # The interface interactive links will activate ["classic", "jupyterlab"]
    binderhub_url             : https://mybinder.org  # The URL of the BinderHub (e.g., https://mybinder.org)
    jupyterhub_url            : ""  # The URL of the JupyterHub (e.g., https://datahub.berkeley.edu)
    thebe                     : false  # Add a thebe button to pages (requires the repository to run on Binder)
    colab_url                 : "" # The URL of Google Colab (https://colab.research.google.com)

    # Information about where the book exists on the web
    repository:
    url: https://github.com/marcdumon/0_jupyter_book  # Online location of your book
    path_to_book: docs  # Optional path to your book, relative to the repository root
    branch: master  # Which branch of the repository should be used when creating links (optional)

    # Add GitHub buttons to your book
    # See https://jupyterbook.org/customize/config.html#add-a-link-to-your-repository
    html:
    use_issues_button: true
    use_repository_button: true

    sphinx:
    config:
        html_js_files:
        - https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js # To make plotly interactive ?
        suppress_warnings:  ["myst.header"] # To surpress "WARNING: Non-consecutive header level increase; 0 to 2 [myst.header]"

    only_build_toc_files: true
    ````
- `_toc.yml`
    ````yaml
    # Table of contents
    # Learn more at https://jupyterbook.org/customize/toc.html

    format: jb-book
    root: intro
    chapters:
    - file: docs/markdown
    - file: docs/notebooks
    - file: docs/markdown-notebooks
    - file: docs/bigrams
    ````
- `_static\custom.css`
    ````
    p {
        text-align: justify;
    }
    .container, .container-lg, .container-md, .container-sm, .container-xl {
        max-width: 1500px;
    }
    ```
- `.gitignore`
    ````
    # Byte-compiled / optimized / DLL files
    __pycache__/
    *.py[cod]
    *$py.class

    # C extensions
    *.so

    # Distribution / packaging
    .Python
    build/
    develop-eggs/
    dist/
    downloads/
    eggs/
    .eggs/
    lib/
    lib64/
    parts/
    sdist/
    var/
    wheels/
    *.egg-info/
    .installed.cfg
    *.egg
    MANIFEST

    # PyInstaller
    #  Usually these files are written by a python script from a template
    #  before PyInstaller builds the exe, so as to inject date/other infos into it.
    *.manifest
    *.spec

    # Installer logs
    pip-log.txt
    pip-delete-this-directory.txt

    # Unit test / coverage reports
    htmlcov/
    .tox/
    .coverage
    .coverage.*
    .cache
    nosetests.xml
    coverage.xml
    *.cover
    .hypothesis/
    .pytest_cache/

    # Translations
    *.mo
    *.pot

    # Django stuff:
    *.log
    local_settings.py
    db.sqlite3

    # Flask stuff:
    instance/
    .webassets-cache

    # Scrapy stuff:
    .scrapy

    # Sphinx documentation
    docs/_build/
    docs/build/
    build/

    # PyBuilder
    target/

    # Jupyter Notebook
    .ipynb_checkpoints

    # pyenv
    .python-version

    # celery beat schedule file
    celerybeat-schedule

    # SageMath parsed files
    *.sage.py

    # Environments
    .env
    .venv
    env/
    venv/
    ENV/
    env.bak/
    venv.bak/

    # Spyder project settings
    .spyderproject
    .spyproject

    # Rope project settings
    .ropeproject

    # mkdocs documentation
    /site

    # mypy
    .mypy_cache/

    # Miscelaneous
    .idea
    .vscode
    *.csv
    *.DS_Store
    *.db
    *.pptx
    *.pkl
    *.h5
    *.zip
    *.tgz
    *.parquet


    # Specific
    _build/
    ````



