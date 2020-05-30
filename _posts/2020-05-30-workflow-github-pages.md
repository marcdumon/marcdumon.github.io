---
layout: page
comments: false
title:  "Workflow Github Pages"
excerpt: "Some guidance to write blog posts on Github Pages"
date:   2021-05-30 08:53:00
mathjax: true
---

This post gives some guidance on which software too use and how to setup a blog on Github Pages.
The OS I use is Linux (Kubuntu 19.10)




### Install software
#### Jekyll
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
- Create and goto a new directory ./myblog
- Create a new Gemfile to list your project’s dependencies:
  ```
  bundle init
  ```
- Edit the Gemfile and add jekyll as a dependency:
  ```
  gem "jekyll"
  ```
- Run bundle to install jekyll for your project:
  ```
  bundle
  ```

#### Github Desktop

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


### Setup Github
- Create a new repository:  
  One you signed in goto [https://github.com/new](https://github.com/new).   
  You have to give your repository a special name to generate your website. That name is *username.github.io* (where “username” is your actual GitHub user name).  
  Fill in this 'Repository name' and click on the 'Create Repository' button.
- Setup your site:   
  Click on 'Settings'   
  ![Settings](/assets/2020-05-30-workflow-github-pages/github_settings.png "Settngs")   
  Scroll down on the settings page, you’ll see the GitHub Pages section near the bottom. Click the Choose a theme button to start the process of creating your site.
  ![Theme](/assets/2020-05-30-workflow-github-pages/github_theme.png 'Theme')   
  Once you’ve clicked the button, you’ll be directed to the Theme Chooser. You’ll see several theme options in a carousel across the top of the page. Click on the images to preview the themes. Once you’ve selected one, click Select theme on the right to move on. It’s easy to change your theme later, so if you’re not sure, just choose one for now.   
  ![Theme Chooser](/assets/2020-05-30-workflow-github-pages/github_theme_chooser.png 'Theme Chooser')   
  You can leave the content for now. Just scroll to the bottom and click on 'Commit changes'
  ![Commit Changes](/assets/2020-05-30-workflow-github-pages/github_commit.png 'Commit Changes')  

### Configure the Website











https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
