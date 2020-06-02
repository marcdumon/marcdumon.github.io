---
layout: page
comments: true
title:  "Setup Github Pages "
excerpt: "Some instructions on how to setup a website on GitHub Pages"
date:   2020-05-30 08:53:00
mathjax: true
---

This post gives some guidance on which software to use and how to setup a website on Github Pages.
The OS I use is Linux (Kubuntu 19.10)

## Install Atom Editor
[Atom Editor](https://atom.io/): Atom is a free and open-source text and source code editor with many plug-ins.

[Instructions](https://flight-manual.atom.io/getting-started/sections/installing-atom/)
- add the package repository to your system:
  ```
  wget -qO - https://packagecloud.io/AtomEditor/atom/gpgkey | sudo apt-key add -
  sudo sh -c 'echo "deb [arch=amd64] https://packagecloud.io/AtomEditor/atom/any/ any main" > /etc/apt/sources.list.d/atom.list'
  sudo apt-get update
  ```
- install Atom using apt:
  ```
  sudo apt install atom
  ```

## Setup Github repository
- Create a new repository:  
  One you signed in go to [https://github.com/new](https://github.com/new).   
  You have to give your repository a special name to generate your website. That name is *username.github.io* (where “username” is your actual GitHub user name).  
  Fill in this 'Repository name' and click on the 'Create Repository' button.
- Setup your site:   
  Click on 'Settings'   
  ![Settings](/assets/2020-05-30-setup-github-pages/github_settings.png "Settings")   
  Scroll down on the settings page, you’ll see the GitHub Pages section near the bottom. Click the Choose a theme button to start the process of creating your site.
  ![Theme](/assets/2020-05-30-setup-github-pages/github_theme.png 'Theme')   
  Once you’ve clicked the button, you’ll be directed to the Theme Chooser. You’ll see several theme options in a carousel across the top of the page. Click on the images to preview the themes. Once you’ve selected one, click Select theme on the right to move on. It’s easy to change your theme later, so if you’re not sure, just choose one for now.   
  ![Theme Chooser](/assets/2020-05-30-setup-github-pages/github_theme_chooser.png 'Theme Chooser')   
  You can leave the content for now. Just scroll to the bottom and click on 'Commit changes'
  ![Commit Changes](/assets/2020-05-30-setup-github-pages/github_commit.png 'Commit Changes')  
- Your github repository now contains the following two files:
  - _config.yml
  - index.html

## Install and setup Github Desktop
[Github Desktop Linux Fork](https://github.com/shiftkey/desktop): Github Desktop is designed to simplify essential steps in your GitHub workflow. It is only available for Windows and Mac. However a Fork of Github Desktop to support various Linux distributions is available from githb/shiftkey.   

[Instructions](https://github.com/shiftkey/desktop):
- Setup the package repository
  ```
  wget -qO - https://packagecloud.io/shiftkey/desktop/gpgkey | sudo apt-key add -
  sudo sh -c 'echo "deb [arch=amd64] https://packagecloud.io/shiftkey/desktop/any/ any main" > /etc/apt/sources.list.d/packagecloud-shiftky-desktop.list'
  sudo apt-get update
  ```
- Install GitHub Desktop:
  ```
  sudo apt install github-desktop
  ```
- Configure Github desktop:   
  Go to <File> <Options> and fill-in your account. On the 'Integration' menu add 'Atom' as 'External Editor' and save
- Clone github repository locally:   
  Click on 'Clone a repository from the Internet'. Select your *username/username.github.io*   
  Choose the local path where you want to save the local repository.   
  Click 'Clone'    
  All changes made on the local repository can be pushed to github now.
- Your github repository now contains the following two files and one hidden directory:
  - _config.yml
  - .git/
  - index.html

## Install and setup Jekyll
[Jekyll](jekyllrb.com): is a static site generator. You give it text written in your favorite markup language and it uses layouts to create a static website. You can tweak how you want the site URLs to look, what data gets displayed on the site, and more.     

[Instructions](https://jekyllrb.com/docs/):   
- Install Ruby development environment:  
  Install dependencies:  
  ```
  sudo apt-get install ruby-full build-essential zlib1g-dev
  ```
  Add environment variables to your ~/.bashrc.
  ```
  echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
  echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
  echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
  ```
- Install Jekyll:
  ```
  gem install jekyll bundler
  ```   

Jekyll is now installed. Let's setup the local repository:
- In terminal go to the local repository *./username.github.io*
- Create a new Gemfile to list your project’s dependencies:
  ```
  bundle init
  ```
  ```> Writing new Gemfile to /home/md/Temp/xxx_blog/Gemfile```
- Your github repository now contains the following three files and one hidden directory:
  - _config.yml
  - .Gemfile.swp
  - Gemfile
  - .git/
  - index.html

- Edit the Gemfile and add Jekyll as a dependency:
  ```
  gem "jekyll"
  ```

- Run bundle to install jekyll and dependencies for your project:
  ```
  bundle
  ```

## Configure the Website
- Edit the _config.yml file:   
  ```yml
  # Site settings
  title:
  email:
  description:
  baseurl: ""
  url: "http://username.github.io"
  twitter_username:
  github_username:  

  # Build settings
  future: true
  markdown: kramdown
  permalink: pretty

  highlighter: rouge
  kramdown:
    input: GFM
    auto_ids: true
    syntax_highlighter: rouge
  ```
- Download the [standard directories](/assets/2020-05-30-setup-github-pages/standard_directories.zip) and copy the directories in your repository directory.
- Change the tracking-id for google analytics in `_includes/head.html`

## Usage
- Start local webserver to see changes:
  In the repository directory:
  ```
  jekyll serve
  ```
- To add a new post, always use the format yyyy-mm-dd-title.md
